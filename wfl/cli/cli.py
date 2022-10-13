"""
Command line interface to the package
"""

import distutils.util
import glob
import json
import os
import sys
import warnings
from pprint import pformat, pprint
from pathlib import Path

import ase.data
import ase.io
import click
import numpy as np
import yaml

try:
    import quippy
except ModuleNotFoundError:
    pass
# noinspection PyProtectedMember
from ase.io.extxyz import key_val_str_to_dict

from wfl.configset import ConfigSet, OutputSpec
from wfl.generate import normal_modes as nm 
import wfl.generate.smiles
import wfl.utils.misc
import wfl.generate.buildcell
import wfl.select.by_descriptor
import wfl.descriptors.quippy

from wfl.utils import gap_xml_tools

from wfl.calculators.dft import evaluate_dft
from wfl.calculators import committee
import wfl.calculators.orca
import wfl.calculators.orca.basinhopping
import wfl.calculators.generic


from wfl.fit import gap as fit_gap
import wfl.fit.ref_error
import wfl.fit.utils


@click.group("wfl")
@click.option("--verbose", "-v", is_flag=True)
@click.pass_context
def cli(ctx, verbose):
    """GAP workflow command line interface.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # ignore calculator writing warnings
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning, module="ase.io.extxyz")


@cli.group("file-op")
@click.pass_context
def subcli_file_operations(ctx):
    pass


@cli.group("processing")
@click.pass_context
def subcli_processing(ctx):
    pass


@cli.group("select-configs")
@click.pass_context
def subcli_select_configs(ctx):
    pass


@cli.group("generate-configs")
@click.pass_context
def subcli_generate_configs(ctx):
    pass


@cli.group("select-configs")
@click.pass_context
def subcli_select_configs(ctx):
    pass


@cli.group("descriptor")
@click.pass_context
def subcli_descriptor(ctx):
    pass


@cli.group("ref-method")
@click.pass_context
def subcli_calculators(ctx):
    pass


@cli.group("fitting")
@click.pass_context
def subcli_fitting(ctx):
    pass



@subcli_generate_configs.command('smiles')
@click.pass_context
@click.argument("smiles", nargs=-1)
@click.option("--output", "-o", help="Output filename, see Configset for details", required=True)
@click.option("--info", "-i", help="Extra info to add to Atoms.info")
def configs_from_smiles(ctx, smiles, output, info):
    """ ase.Atoms from SMILES string"""

    verbose = ctx.obj["verbose"]

    outputspec = OutputSpec(output)

    if info is not None:
        info = key_val_str_to_dict(info)

    if verbose:
        print(f'smiles: {smiles}')
        print(f'info: {info}')
        print(outputspec)

    wfl.generate.smiles.run(smiles, outputs=outputspec, extra_info=info)


@subcli_file_operations.command("gather")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output", "-o", help="Output filename, see Configset for details", required=True)
@click.option("--index", "-i", type=click.STRING, required=False,
              help="Pass this index to configset globally")
@click.option("--force", "-f", help="force writing", is_flag=True)
def file_gather(ctx, inputs, output, force, index):
    """ Gathers configurations from files through a Configset
    """
    verbose = ctx.obj["verbose"]

    # ignore calculator writing warnings
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning, module="ase.io.extxyz")

    configset = ConfigSet(inputs)
    outputspec = OutputSpec(output)

    if verbose:
        print(configset)
        print(outputspec)

    for at in configset:
        outputspec.store(at, at.info.pop("_ConfigSet_loc"))
    outputspec.close()


@subcli_file_operations.command("strip")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--keep-info", "-i", required=False, multiple=True,
              help="Keys to keep from info if present")
@click.option("--keep-array", "-a", required=False, multiple=True,
              help="Keys to keep from arrays if present")
@click.option("--cell", is_flag=True, help="Keep the cell w/ PBC")
@click.option("--output", "-o", help="Output filename, see Configset for details",
              type=click.STRING, required=False)
@click.option("--force", "-f", help="force writing", is_flag=True)
def strip(ctx, inputs, keep_info, keep_array, cell, output, force):
    """Strips structures of all info and arrays except the ones specified to keep

    Notes
    -----
    can be replaced by a call on `ase convert` when that allows for taking None of the
    info/arrays keys
    see issue: https://gitlab.com/ase/ase/-/issues/727
    """
    verbose = ctx.obj["verbose"]

    if output is None:
        if not force:
            raise ValueError(
                "Error in: `wfl file-op strip`: neither output nor force are given, specify one "
                "at least")
        output = inputs

    configset = ConfigSet(inputs)
    outputspec = OutputSpec(output)

    # iterate, used for both progressbar and without the same way
    for at in configset:
        new_at = ase.Atoms(at.get_chemical_symbols(), positions=at.get_positions())

        if cell:
            new_at.set_cell(at.get_cell())
            new_at.set_pbc(at.get_pbc())

        if keep_info is not None:
            for key, val in at.info.items():
                if key in keep_info:
                    new_at.info[key] = val

        if keep_array is not None:
            for key, val in at.arrays.items():
                if key in keep_array:
                    new_at.arrays[key] = val

        outputspec.store(new_at, at.info.get("_ConfigSet_loc"))

    outputspec.close()


@subcli_processing.command("committee")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--prefix", "-p", help="Prefix to put on the first", type=click.STRING,
              default="committee.")
@click.option("--gap-fn", "-g", type=click.STRING, help="gap filenames, globbed", required=True,
              multiple=True)
# @click.option("--stride", "-s", "-n", help="Take every Nth frame only", type=click.STRING,
#               required=False)
@click.option("--force", "-f", help="force writing", is_flag=True)
def calc_ef_committee(ctx, inputs, prefix, gap_fn, force):
    """Calculated energy and force with a committee of models.
    Uses a prefix on the filenames, this can be later changed
    does not support globbed filenames as xyz input

    configset works in many -> many mode here
    """

    verbose = ctx.obj["verbose"]

    # apply the prefix to output names
    outputs = {fn: os.path.join(os.path.dirname(fn), f"{prefix}{os.path.basename(fn)}") for fn in
               inputs}

    configset = ConfigSet(inputs)
    outputspec = OutputSpec(outputs)

    if verbose:
        print(configset)
        print(outputspec)

    # read GAP models
    gap_fn_list = []
    for fn in gap_fn:
        gap_fn_list.extend(sorted(glob.glob(fn)))
    gap_model_list = [(quippy.potential.Potential, "", dict(param_filename=fn)) for fn in gap_fn_list]

    # calculate E,F
    for at in configset:
        at_out = committee.calculate_committee(at, gap_model_list)
        outputspec.store(at_out, at.info.get["_ConfigSet_loc"])

    outputspec.close()



@subcli_generate_configs.command("repeat-buildcell")
@click.pass_context
@click.option("--output-file", type=click.STRING)
@click.option("--buildcell-input", type=click.STRING, required=True, help="buildcell input file")
@click.option("--buildcell-exec", type=click.STRING, required=True,
              help="buildcell executable including path")
@click.option("--n-configs", "-N", type=click.INT, required=True,
              help="number of configs to generate")
@click.option("--extra-info", type=click.STRING, default="",
              help="dict of information to store in Atoms.info")
@click.option("--perturbation", type=click.FLOAT, default=0.0,
              help="magnitude of random perturbation to atomic positions")
def _repeat_buildcell(ctx, output_file, buildcell_input, buildcell_exec,
                      n_configs,
                      extra_info, perturbation):
    """Repeatedly runs buildcell (from Pickard's AIRSS distribution) to generate random configs with
    specified species, volumes, distances, symmetries, etc.

    Minimal contents of --buildcell-input file:

    \b
    #TARGVOL=<min_vol>-<max_vol> (NOTE: volume is volume_per_formula_unit/number_of_species)
    #SPECIES=<elem_symbol_1>%NUM=<num_1>[,<elem_symbol_2>%NUM=<num_2 ...]
    #NFORM=[ <n_min>-<n_max> | { <n_1>, <n_2>, ... } ]
    #SYMMOPS=<n_min>-<n_max> (NOTE: optional)
    #SLACK=0.25
    #OVERLAP=0.1
    #COMPACT
    #MINSEP=<min_separation_default> <elem_symbol_1>-<elem_symbol_1>=<min_separation_1_1> [
    <elem_symbol_1>-<elem_symbol_2=<min_separation_1_2> ... ]
    ##EXTRA_INFO <info_field>=<value> (NOTE: optional)
    """
    extra_info = key_val_str_to_dict(extra_info)
    with open(buildcell_input) as bc_f:
        buildcell_input_txt = bc_f.read()

    wfl.generate.buildcell.run(
        outputs=OutputSpec(output_file),
        config_is=range(n_configs),
        buildcell_cmd=buildcell_exec,
        buildcell_input=buildcell_input_txt,
        extra_info=extra_info,
        perturbation=perturbation,
        verbose=ctx.obj["verbose"]
    )


@subcli_select_configs.command("CUR-global")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--n-configs", "-N", type=click.INT, required=True,
              help="number of configs to select")
@click.option("--descriptor-key", type=click.STRING, help="Atoms.info key for descriptor vector")
@click.option("--descriptor", type=click.STRING, help="quippy.Descriptor arg string")
@click.option("--keep_descriptor", is_flag=True, help="keep the descriptor value in the final config file")
@click.option("--kernel_exponent", type=click.FLOAT, help="exponent of dot-product for kernel")
@click.option("--deterministic", is_flag=True,
              help="use deterministic (not stochastic) CUR selection")
def _CUR_global(ctx, inputs, output_file, n_configs,
                descriptor_key, descriptor, keep_descriptor,
                kernel_exponent, deterministic):
    if descriptor is None:
        if descriptor_key is None:
            raise RuntimeError('CUR-global needs --descriptor or --descriptor-key')
    clean_tmp_files = False
    if descriptor is not None:
        # calculate descriptor
        if descriptor_key is None:
            descriptor_key = '_CUR_desc'
        _do_calc_descriptor(inputs, '_tmp_desc.xyz', descriptor, descriptor_key, local=False, force=True)
        inputs = ['_tmp_desc.xyz']
        clean_tmp_files = True

    wfl.select.by_descriptor.CUR_conf_global(
        inputs=ConfigSet(inputs),
        outputs=OutputSpec(output_file),
        num=n_configs,
        at_descs_info_key=descriptor_key, kernel_exp=kernel_exponent, stochastic=not deterministic,
        keep_descriptor_info=keep_descriptor)

    if clean_tmp_files:
        for input_file in inputs:
            Path(input_file).unlink()


@subcli_descriptor.command("calc")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--descriptor", type=click.STRING, required=True, help="quippy.Descriptor arg string")
@click.option("--key", type=click.STRING, required=True,
              help="key to store in Atoms.info (global) or Atoms.arrays(local)")
@click.option("--local", is_flag=True, help="calculate a local (per-atom) descriptor")
@click.option("--force", is_flag=True, help="overwrite existing info or arrays item if present")
def _calc_descriptor(ctx, inputs, output_file, descriptor, key, local, force):
    _do_calc_descriptor(inputs, output_file, descriptor, key, local, force)


def _do_calc_descriptor(inputs, output_file, descriptor, key, local, force):
    wfl.descriptors.quippy.calc(
        inputs=ConfigSet(inputs),
        outputs=OutputSpec(output_file),
        descs=descriptor,
        key=key,
        local=local,
        force=force
    )


@subcli_calculators.command("vasp-eval")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--base-rundir", type=click.STRING, help='directory to run jobs in')
@click.option("--directory-prefix", type=click.STRING, default='run_VASP_')
@click.option("--output-prefix", type=click.STRING)
@click.option("--properties", type=click.STRING, default='energy,forces,stress')
@click.option("--incar", type=click.STRING, help='INCAR file, optional')
@click.option("--kpoints", type=click.STRING, help='KPOINTS file, optional')
@click.option("--vasp-kwargs", type=click.STRING, default="isym=0 isif=7 nelm=300 ediff=1.0e-7",
              help='QUIP-style key-value pairs for ASE vasp calculator kwargs that override contents of INCAR and KPOINTS if both are provided.'
                   '"pp", which is normallly XC-based dir to put between VASP_PP_PATH and POTCAR dirs defaults to ".". Key VASP_PP_PATH will be '
                   'used to set corresponding env var, which is used as dir above <chem_symbol>/POTCAR')
@click.option("--vasp-command", type=click.STRING)
def _vasp_eval(ctx, inputs, output_file, workdir_root, directory_prefix,
               output_prefix, properties,
               incar, kpoints, vasp_kwargs, vasp_command):
    vasp_kwargs = key_val_str_to_dict(vasp_kwargs)
    vasp_kwargs['INCAR_file'] = incar
    vasp_kwargs['KPOINTS_file'] = kpoints
    evaluate_dft(
        inputs=ConfigSet(inputs),
        outputs=OutputSpec(output_file),
        calculator_name="VASP",
        workdir_root=workdir_root,
        dir_prefix=directory_prefix,
        output_prefix=output_prefix,
        properties=properties.split(','),
        calculator_kwargs=vasp_kwargs,
        calculator_command=vasp_command)

    
@subcli_calculators.command("castep-eval")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--output-prefix", type=click.STRING, help="prefix in info/arrays for results")
@click.option("--base-rundir", type=click.STRING, help="directory to put all calculation directories into")
@click.option("--directory-prefix", type=click.STRING, default='run_CASTEP_')
@click.option("--properties", type=click.STRING, default='energy forces stress',
              help="properties to calculate, string is split")
@click.option("--castep-command", type=click.STRING, help="command, including appropriate mpirun")
@click.option("--castep-kwargs", type=click.STRING, help="CASTEP keywords, passed as dict")
@click.option("--keep-files", type=click.STRING, default="default",
              help="How much of files to keep, default is NOMAD compatible subset")
def _castep_eval(ctx, inputs, output_file, workdir_root, directory_prefix, properties,
                 castep_command, castep_kwargs, keep_files, output_prefix):
    if castep_kwargs is not None:
        castep_kwargs = key_val_str_to_dict(castep_kwargs)

    evaluate_dft(
        inputs=ConfigSet(inputs),
        outputs=OutputSpec(output_file),
        calculator_name="CASTEP",
        workdir_root=workdir_root,
        dir_prefix=directory_prefix,
        properties=properties.split(),
        calculator_command=castep_command,
        keep_files=keep_files,
        calculator_kwargs=castep_kwargs,
        output_prefix=output_prefix
    )


@subcli_calculators.command("orca-eval-basin-hopping")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--output-prefix", type=click.STRING, help="prefix in info/arrays for results")
@click.option("--base-rundir", type=click.STRING,
              help="directory to put all calculation directories into")
@click.option("--directory-prefix", type=click.STRING, default='ORCA')
@click.option("--calc-kwargs", "--kw", type=click.STRING, required=False,
              default=None,
              help="Kwargs for calculation, overwritten by other options")
@click.option("--keep-files", type=click.STRING, default="default",
              help="How much of files to keep, default is NOMAD compatible subset")
@click.option("--orca-command", type=click.STRING, help="path to ORCA executable, default=`orca`")
@click.option("--scratch-path", "-tmp", type=click.STRING,
              help="Directory to use as scratch for calculations, SSD recommended, default: cwd")
@click.option("--n-run", "-nr", type=click.INT, required=True,
              help="Number of global optimisation runs for each frame")
@click.option("--n-hop", "-nh", type=click.INT, required=True,
              help="Number of hopping steps to take per run")
@click.option("--orca-simple-input", type=click.STRING,
              help="orca simple input line, make sure it is correct, default "
                   "is recPBE with settings tested for radicals")
@click.option("--orca-additional-blocks", type=click.STRING,
              help="orca blocks to be added, default is None")
def orca_eval_bh(ctx, inputs, workdir_root, output_file, directory_prefix,
              orca_command, calc_kwargs, keep_files, output_prefix, scratchdir, n_run, n_hop,
              orca_simple_input, orca_additional_blocks):
    verbose = ctx.obj["verbose"]

    if scratchdir is not None:
        if not os.path.isdir(scratchdir):
            raise NotADirectoryError(
                f"Scratch path needs to be a directory, invalid given: {scratchdir}")
        if not os.access(scratchdir, os.W_OK):
            raise PermissionError(f"cannot write to specified scratch dir: {scratchdir}")
        scratchdir = os.path.abspath(scratchdir)

    try:
        keep_files = bool(distutils.util.strtobool(keep_files))
    except ValueError:
        if keep_files != 'default':
            raise RuntimeError(f'invalid value given for "keep_files" ({keep_files})')

    # default: dict()
    if calc_kwargs is None:
        calc_kwargs = dict()
    else:
        calc_kwargs = key_val_str_to_dict(calc_kwargs)

    # update args
    for key, val in dict(orca_command=orca_command, scratchdir=scratchdir, n_run=n_run,
                         n_hop=n_hop, orcasimpleinput=orca_simple_input,
                         orcablock=orca_additional_blocks).items():
        if val is not None:
            calc_kwargs[key] = val

    configset = ConfigSet(inputs)
    outputspec = OutputSpec(output_file)

    if verbose:
        print(configset)
        print(outputspec)
        print("ORCA wfn-basin hopping calculation parameters: ", calc_kwargs)

    wfl.calculators.orca.basinhopping.evaluate_basin_hopping(
        inputs=configset, outputs=outputspec, workdir_root=workdir_root, dir_prefix=directory_prefix,
        keep_files=keep_files, output_prefix=output_prefix, orca_kwargs=calc_kwargs
    )


@subcli_calculators.command("orca-eval")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--output-prefix", type=click.STRING, help="prefix in info/arrays for results")
@click.option("--workdir", type=click.STRING, help="directory to put all calculation directories into")
@click.option("--rundir-prefix", type=click.STRING, default='ORCA_')
@click.option("--keep-files", type=click.STRING, default="default",
              help="How much of files to keep, default is NOMAD compatible subset")
@click.option("--calculator-exec", type=click.STRING, help="path to ORCA executable, default=`orca`")
@click.option("--scratchdir", "-tmp", type=click.STRING,
              help="Directory to use as scratch for calculations, SSD recommended, default: cwd")
@click.option("--orcasimpleinput", type=click.STRING, help="orca simple input line, make sure it is correct, default "
                                                             "is recPBE with settings tested for radicals")
@click.option("--orcablocks", type=click.STRING, help="orca blocks to be added, default is None")
@click.option("--charge", type=click.INT, default=0, help='charge for the calculation')
@click.option("--mult", type=click.INT, default=None, help='multiplicity')
def orca_eval(ctx, inputs, output_file, output_prefix, workdir,
              rundir_prefix, keep_files, calculator_exec, scratchdir,
              orcasimpleinput, orcablocks, charge, mult):

    verbose = ctx.obj["verbose"]

    if scratchdir is not None:
        if not os.path.isdir(scratchdir):
            raise NotADirectoryError(f"Scratch path needs to be a directory, invalid given: {scratchdir}")
        if not os.access(scratchdir, os.W_OK):
            raise PermissionError(f"cannot write to specified scratch dir: {scratchdir}")
        scratchdir = os.path.abspath(scratchdir)

    try:
        keep_files = bool(distutils.util.strtobool(keep_files))
    except ValueError:
        if keep_files != 'default':
            raise RuntimeError(f'invalid value given for "keep_files" ({keep_files})')


    configset = ConfigSet(inputs)
    outputspec = OutputSpec(output_file)

    orca_params = {
        "keep_files": keep_files,
        "rundir_prefix": rundir_prefix,
        "workdir": workdir,
        "scratchdir": scratchdir,
        "calculator_exec": calculator_exec,
        "charge": charge,
        "mult": mult
    }

    if orcasimpleinput is not None:
        orca_params["orcasimpleinput"] = orcasimpleinput
    if orcablocks is not None:
        orca_params["orcablocks"] = orcablocks

    if verbose:
        print(configset)
        print(outputspec)
        print(f"orca params: {orca_params}")
    
    calc = (wfl.calculators.orca.ORCA, [], orca_params)

    wfl.calculators.generic.run(
        inputs=configset, 
        outputs=outputspec,
        calculator=calc, 
        properties=["energy", "forces", "dipole"],
        output_prefix=output_prefix,
    )


@subcli_processing.command("reference-error")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--pre-calc", "-pre", type=click.STRING, required=True,
              help="string to exec() that sets up for calculator, e.g. imports")
@click.option("--calc", "-c", type=click.STRING, required=True,
              help="string to 'eval()' that returns calculator constructor")
@click.option("--calc-args", "-a", type=click.STRING,
              help="json list of calculator constructor args", default="[]")
@click.option("--calc-kwargs", "-k", type=click.STRING,
              help="json dict of calculator constructor kwargs", default="{}")
@click.option("--ref-prefix", "-r", type=click.STRING,
              help="string to prepend to info/array keys for reference energy, "
                   "forces, virial. If None, use SinglePointCalculator results")
@click.option("--properties", "-p", type=click.STRING,
              help="command separated list of properties to use",
              default='energy_per_atom,forces,virial_per_atom')
@click.option("--category_keys", "-C", type=click.STRING,
              help="comma separated list of info keys to use for doing per-category error",
              default="")
@click.option("--outfile", "-o", type=click.STRING, help="output file, - for stdout", default='-')
@click.option("--intermed-file", type=click.STRING,
              help="intermediate file to contain calculator results, keep in memory if None")
def ref_error(ctx, inputs, pre_calc, calc, calc_args, calc_kwargs, ref_prefix, properties,
              category_keys, outfile, intermed_file):
    verbose = ctx.obj["verbose"]

    cs_out = OutputSpec(intermed_file)

    if pre_calc is not None:
        exec(pre_calc)

    if ref_prefix is None:
        # copy from SinglePointCalculator to info/arrays so calculator results won't overwrite
        # will do this by keeping copy of configs in memory, maybe should have an optional way to do
        # this via a file instead.
        inputs = list(ConfigSet(inputs))
        ref_property_keys = wfl.fit.utils.copy_properties(inputs, ref_property_keys=ref_prefix)
        inputs = ConfigSet(inputs)
    else:
        ref_property_keys = {p: ref_prefix + p for p in
                             ['energy', 'forces', 'stress', 'virial']}
        inputs = ConfigSet(inputs)

    errs = wfl.fit.ref_error.calc(inputs, cs_out,
                                  calculator=(
                                  eval(calc), json.loads(calc_args), json.loads(calc_kwargs)),
                                  ref_property_keys=ref_property_keys,
                                  properties=[p.strip() for p in properties.split(',')],
                                  category_keys=category_keys.split(', '))

    if outfile == '-':
        pprint(errs)
    else:
        with open(outfile, 'w') as fout:
            fout.write(pformat(errs) + '\n')


@subcli_fitting.command("multistage-gap")
@click.pass_context
@click.argument("inputs", nargs=-1, required=True)
@click.option("--GAP-name", "-G", type=click.STRING, required=True,
              help="name of final GAP file, not including xml suffix")
@click.option("--params-file", "-P", type=click.STRING, required=True,
              help="fit parameters JSON file")
@click.option("--property_prefix", "-p", type=click.STRING,
              help="prefix to reference property keys")
@click.option("--database-modify-mod", "-m", type=click.STRING,
              help="python module that defines a 'modify()' function for operations like setting "
                   "per-config fitting error")
@click.option("--run-dir", "-d", type=click.STRING, help="subdirectory to run in")
@click.option("--fitting-error/--no-fitting-error", help="calculate error for fitting configs",
              default=True)
@click.option("--testing-configs", type=click.STRING,
              help="space separated list of files with testing configurations to calculate error for")
def multistage_gap(ctx, inputs, gap_name, params_file, property_prefix, database_modify_mod,
                   run_dir, fitting_error,
                   testing_configs):
    verbose = ctx.obj["verbose"]

    with open(params_file) as fin:
        fit_params = json.load(fin)

    if testing_configs is not None:
        testing_configs = ConfigSet(testing_configs.split())

    GAP, fit_err, test_err = fit_gap.multistage.fit(ConfigSet(inputs),
                                                    GAP_name=gap_name, params=fit_params,
                                                    ref_property_prefix=property_prefix,
                                                    database_modify_mod=database_modify_mod,
                                                    calc_fitting_error=fitting_error,
                                                    testing_configs=testing_configs,
                                                    run_dir=run_dir, verbose=verbose)


@subcli_fitting.command('simple-gap')
@click.pass_context
@click.option('--gap-file', '-g', default='GAP.xml', show_default=True,
              help='GAP filename, overrides option'
                   'in parameters file')
@click.option('--atoms-filename',
              help='xyz with training configs and isolated atoms')
@click.option('--param-file', '-p', required=True,
              help='yml file with gap parameters ')
@click.option('--gap-fit-command', default='gap_fit',
              help='executable for gap_fit')
@click.option('--output-file', '-o', default='default',
              help='filename where to save gap output, defaults to '
                   'gap_basename + _output.txt')
@click.option('--fit', is_flag=True, help='Actually run the gap_fit command')
@click.option("--verbose", "-v", is_flag=True)
def simple_gap_fit(ctx, gap_file, atoms_filename, param_file,
                   gap_fit_command, output_file, fit, verbose):
    """Fit a GAP with descriptors from  an .yml file"""

    # read properties from the param file
    with open(param_file) as yaml_file:
        params = yaml.safe_load(yaml_file)

    if atoms_filename is None:
        # not on command line, try in params
        if 'atoms_filename' in params:
            atoms_filename = params['atoms_filename']
        else:
            raise RuntimeError('atoms_filename not given in params file or as '
                               'command line input')
    else:
        if 'atoms_filename' in params:
            raise RuntimeError('atoms_filename given in params file and as '
                               'command line input')

    fitting_ci = ConfigSet(atoms_filename)

    if gap_file != 'GAP.xml':
        if params.get('gap_file', False):
            warnings.warn('overwriting gap_file from params with value from '
                          'command line')
        params['gap_file'] = gap_file

        if verbose:
            print("Overwritten param file's gap-filename from command")

    if output_file == 'default':
        output_file = os.path.splitext(params['gap_file'])[0] + '_output.txt'

    fit_gap.simple.run_gap_fit(fitting_ci, fitting_dict=params,
                               stdout_file=output_file,
                               gap_fit_command=gap_fit_command,
                               do_fit=fit, verbose=verbose)


if __name__ == '__main__':
    cli()
