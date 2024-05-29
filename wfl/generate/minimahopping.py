import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.optimize.minimahopping import MinimaHopping
from ase.io.trajectory import Trajectory
import ase.io

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.misc import atoms_to_list
from wfl.generate.utils import config_type_append
from wfl.utils.parallel import construct_calculator_picklesafe



def _get_MD_trajectory(rundir):

    md_traj = []
    mdtrajfiles = sorted([file for file in Path(rundir).glob("md*.traj")])
    for mdtraj in mdtrajfiles:
        for at in ase.io.read(f"{mdtraj}", ":"):
            config_type_append(at, 'traj')
            md_traj.append(at)

    return md_traj


# perform MinimaHopping on one ASE.atoms object
def _atom_opt_hopping(atom, calculator, Ediff0, T0, minima_threshold, mdmin, minima_traj,
                     fmax, timestep, totalsteps, maxtemp, skip_failures, workdir=None, **opt_kwargs):
    save_tmpdir = opt_kwargs.pop("save_tmpdir", False)
    return_all_traj = opt_kwargs.pop("return_all_traj", False)
    origdir = Path.cwd()
    if workdir is None:
        workdir = Path.cwd()
    else:
        workdir = Path(workdir)

    rundir = tempfile.mkdtemp(dir=workdir, prefix='Opt_hopping_')

    os.chdir(rundir)
    atom.calc = calculator
    try:
        opt = MinimaHopping(atom, Ediff0=Ediff0, T0=T0, minima_threshold=minima_threshold,
                            mdmin=mdmin, fmax=fmax, timestep=timestep, **opt_kwargs)
        opt(totalsteps=totalsteps, maxtemp=maxtemp)
    except Exception as exc:
        # optimization may sometimes fail to converge.
        if skip_failures:
            sys.stderr.write(f'Structure optimization failed with exception \'{exc}\'\n')
            sys.stderr.flush()
            os.chdir(workdir)
            shutil.rmtree(rundir)
            os.chdir(origdir)
            return None
        else:
            raise
    else:
        traj = []
        if return_all_traj:
            traj += _get_MD_trajectory(rundir)

        for hop_traj in Trajectory(minima_traj):
            config_type_append(hop_traj, 'minima')
            traj.append(hop_traj)
        if not save_tmpdir:
            os.chdir(workdir)
            shutil.rmtree(rundir)
        os.chdir(origdir)
        return traj

    os.chdir(origdir)


def _run_autopara_wrappable(atoms, calculator, Ediff0=1, T0=1000, minima_threshold=0.5, mdmin=2, minima_traj="minima.traj",
                           fmax=0.05, timestep=1, totalsteps=10, maxtemp=None, skip_failures=True, workdir=None,
                           rng=None, _autopara_per_item_info=None,
                           **opt_kwargs):
    """runs a structure optimization

    Parameters
    ----------
    atoms: list(Atoms)
        input configs
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    Ediff0: float, default 1 (eV)
        initial energy acceptance threshold
    T0: float, default 1000 (K)
        initial MD temperature
    minima_threshold: float, default 0.5 (A)
        threshold for identical configs
    mdmin: int, default 2
        criteria to stop MD simulation (number of minima)
    fmax: float, default 1 (eV/A)
        max force for optimizations
    timestep: float, default 1 (fs)
        timestep for MD simulations
    totalsteps: int, default 10
        number of steps
    skip_failures: bool, default True
        just skip optimizations that raise an exception
    workdir: str/Path default None
        workdir for saving files
    opt_kwargs
        keyword arguments for MinimaHopping
    rng: numpy.random.Generator, default None
        random number generator to use (needed for pressure sampling, initial temperature, or Langevin dynamics)
    _autopara_per_item_info: dict
        INTERNALLY used by autoparallelization framework to make runs reproducible (see
        wfl.autoparallelize.autoparallelize() docs)

    Returns
    -------
        list(Atoms) trajectories
    """

    calculator = construct_calculator_picklesafe(calculator)
    all_trajs = []

    for at_i, at in enumerate(atoms_to_list(atoms)):
        if _autopara_per_item_info is not None:
            # minima hopping doesn't let you pass in a np.random.Generator, so set a global seed using
            # current generator
            np.random.seed(_autopara_per_item_info[at_i]["rng"].integers(2 ** 32))

        traj = _atom_opt_hopping(atom=at, calculator=calculator, Ediff0=Ediff0, T0=T0, minima_threshold=minima_threshold,
                                 mdmin=mdmin, fmax=fmax, timestep=timestep, totalsteps=totalsteps, minima_traj=minima_traj, maxtemp=maxtemp,
                                 skip_failures=skip_failures, workdir=workdir, **opt_kwargs)
        all_trajs.append(traj)

    return all_trajs


# run that operation on ConfigSet, for multiprocessing
def minimahopping(*args, **kwargs):
    default_autopara_info = {"num_inputs_per_python_subprocess": 10}

    return autoparallelize(_run_autopara_wrappable, *args,
                           default_autopara_info=default_autopara_info, **kwargs)
autoparallelize_docstring(minimahopping, _run_autopara_wrappable, "Atoms")
