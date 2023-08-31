import numpy as np
import os, sys, json, yaml, subprocess
from expyre import ExPyRe
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize.utils import get_remote_info
from expyre.resources import Resources
from pathlib import Path 
from shutil import copyfile



#def get_mace(params_file):
#	"""
#	Run the wfl.fit.gap_multistage fit function.
#
#	Parameters
#	----------
#	in_file:                str
#		Path to file containing the input configs for the GAP fit
#	gap_name:               str
#		File name of written GAP
#	Zs:                     list
#		List of atomic numbers in the GAP fit.
#	length_scales:          dict
#		Length scale dictionary for each atomic species in the fit.
#		Dictionary keys are the atomic numbers, values are dictionaries that
#		must contain the keys "bond_len" and "min_bond_len"
#	params:                 dict
#		GAP fit parameters, see the parameter json files for more information
#	ref_property_prefix:    str, default="DFT_"
#		label prefixes for the in_config properties.
#	run_dir:                str, default='GAP'
#		Name of the directory in which the GAP files will be written
#	Returns
#	-------
#	None, the selected configs are written in the out_file
#	"""
#
#	params = yaml.safe_load(Path(params_file).read_text())
#
#	for key, val in params.items():
#		if isinstance(val, int) or isinstance(val, float):
#			mace_fit_cmd += f" --{key}={val}"
#		elif isinstance(val, str):
#			mace_fit_cmd += f" --{key}='{val}'"
#		elif val is None:
#			mace_fit_cmd += f" --{key}"
#		else:
#			mace_fit_cmd += f" --{key}='{val}'"
#	
#	try:
#		subprocess.run(mace_fit_cmd, shell=True, check=True)
#	except subprocess.CalledProcessError as e:
#		print("Failure in calling GAP fitting with error code:", e.returncode)
#		raise e
#	
#	return None


def run_mace_fit(params, mace_name="mace", run_dir=".", remote_info=None, mace_fit_cmd="python  ~/Softwares/mace/scripts/run_train.py",
		verbose=True, do_fit=True, wait_for_results=True, remote_label=None, skip_if_present=True, **kwargs):

	run_dir = Path(run_dir)

	if skip_if_present:
		try:
			print(f"check whether already fitted model exists as {run_dir}/{mace_name}.model")
			if not Path(f"{run_dir}/{mace_name}.model").is_file():
				raise FileNotFoundError

			return mace_name
		except (FileNotFoundError, RuntimeError):
			pass


	if remote_info != '_IGNORE':
		remote_info = get_remote_info(remote_info, remote_label)
#	print("remote_info : ", remote_info)

	if remote_info is not None and remote_info != '_IGNORE':
		input_files = remote_info.input_files.copy()
		output_files = remote_info.output_files.copy() + [str(run_dir)]


		# set number of threads in queued job, only if user hasn't set them
		if not any([var.split('=')[0] == 'WFL_GAP_FIT_OMP_NUM_THREADS' for var in remote_info.env_vars]):
			remote_info.env_vars.append('WFL_GAP_FIT_OMP_NUM_THREADS=$EXPYRE_NUM_CORES_PER_NODE')
		if not any([var.split('=')[0] == 'WFL_NUM_PYTHON_SUBPROCESSES' for var in remote_info.env_vars]):
			remote_info.env_vars.append('WFL_NUM_PYTHON_SUBPROCESSES=$EXPYRE_NUM_CORES_PER_NODE')

		remote_func_kwargs = {'params': params,'remote_info': '_IGNORE', 'run_dir': run_dir,
							'input_files' : remote_info.input_files.copy()}

		kwargs.update(remote_func_kwargs)
		xpr = ExPyRe(name=remote_info.job_name, pre_run_commands=remote_info.pre_cmds, post_run_commands=remote_info.post_cmds,
					 env_vars=remote_info.env_vars, input_files=input_files, output_files=output_files, function=run_mace_fit,
					 kwargs = remote_func_kwargs)

		xpr.start(resources=remote_info.resources, system_name=remote_info.sys_name, header_extra=remote_info.header_extra,
				  exact_fit=remote_info.exact_fit, partial_node=remote_info.partial_node)

		if not wait_for_results:
			return None
		results, stdout, stderr = xpr.get_results(timeout=remote_info.timeout, check_interval=remote_info.check_interval)

		sys.stdout.write(stdout)
		sys.stderr.write(stderr)

		# no outputs to rename since everything should be in run_dir
		xpr.mark_processed()
		
		return results

	if not run_dir.exists():
		run_dir.mkdir(parents=True)

	if isinstance(params, str):
		params = yaml.safe_load(Path(params).read_text())
	elif isinstance(params, dict):
		pass

	for key, val in params.items():
		if isinstance(val, int) or isinstance(val, float):
			mace_fit_cmd += f" --{key}={val}"
		elif isinstance(val, str):
			mace_fit_cmd += f" --{key}='{val}'"
		elif val is None:
			mace_fit_cmd += f" --{key}"
		else:
			mace_fit_cmd += f" --{key}='{val}'"
	
	if not do_fit or verbose:
		print('fitting command:\n', mace_fit_cmd)


	orig_omp_n = os.environ.get('OMP_NUM_THREADS', None)
	if 'WFL_GAP_FIT_OMP_NUM_THREADS' in os.environ:
		os.environ['OMP_NUM_THREADS'] = os.environ['WFL_GAP_FIT_OMP_NUM_THREADS']
	
	try:
		remote_cwd = os.getcwd()	
		if str(run_dir) != ".":
			for input_file in kwargs["input_files"]:
				file_name = input_file.split("/")[-1]
#				copyfile(input_file, f"{run_dir}/{input_file}")
				copyfile(input_file, f"{run_dir}/{file_name}")
			os.chdir(run_dir)
			subprocess.run(mace_fit_cmd, shell=True, check=True)
			os.chdir(remote_cwd)	
		else:
			subprocess.run(mace_fit_cmd, shell=True, check=True)

	except subprocess.CalledProcessError as e:
		print("Failure in calling MACE fitting with error code:", e.returncode)
		raise e
	

if __name__ == "__main__":
	cwd = os.getcwd()
	run_mace_fit(f"{cwd}/params.yaml", remote_info = remote_info)
#	run_mace_fit(f"{cwd}/params.yaml")

