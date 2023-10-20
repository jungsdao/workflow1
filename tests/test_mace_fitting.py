import yaml, os
import time
from pathlib import Path
import mace, json
from wfl.fit.mace import fit
from wfl.configset import ConfigSet
import ase.io
import pytest

# This part is used to detect the location of 'run_train.py' file. 
#mace_dist_dir = list(Path(mace.__file__).parent.parent.glob("mace-*"))[0]
#jsonfile = str(Path(mace_dist_dir / "direct_url.json"))
#with open(jsonfile) as f:
#    d = json.load(f)
#script_dir = d["url"].split(":")[-1][2:]
#mace_fit_cmd = f"python {script_dir}/scripts/run_train.py"
#os.environ["WFL_MACE_FIT_COMMAND"] = "python /raven/u/hjung/Softwares/mace/scripts/run_train.py"

@pytest.mark.skipif(not os.environ.get("WFL_MACE_FIT_COMMAND"), reason="WFL_MACE_FIT_COMMAND not in environment variables")
def test_mace_fit_from_list(request, tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)
    print('getting fitting data from ', request.path)
    print("here :  ", request.path)
	
    parent_path = request.path.parent
    params_file_path = parent_path / 'assets' / 'mace_fit_parameters.yaml'
    mace_fit_params = yaml.safe_load(params_file_path.read_text())
    filepath = parent_path / 'assets' / 'B_DFT_data.xyz'
    fitting_configs = ConfigSet(ase.io.read(filepath, ":"))

    t0 = time.time()
    fit(fitting_configs, "test", mace_fit_params, mace_fit_cmd=None, run_dir=".")
    t_run = time.time() - t0

    assert (tmp_path / "test.model").exists()
    assert (tmp_path / "test.model").stat().st_size > 0

@pytest.mark.skipif(not os.environ.get("WFL_MACE_FIT_COMMAND"), reason="WFL_MACE_FIT_COMMAND not in environment variables")
def test_mace_fit(request, tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)
    print('getting fitting data from ', request.path)
    print("here :  ", request.path)

    parent_path = request.path.parent
    fit_config_file = parent_path / 'assets' / 'B_DFT_data.xyz'
    params_file_path = parent_path / 'assets' / 'mace_fit_parameters.yaml'
    mace_fit_params = yaml.safe_load(params_file_path.read_text())
    fitting_configs = ConfigSet(fit_config_file) 

    t0 = time.time()
    fit(fitting_configs, "test", mace_fit_params, mace_fit_cmd=None, run_dir=".")
    t_run = time.time() - t0

    assert (tmp_path / "test.model").exists()
    assert (tmp_path / "test.model").stat().st_size > 0


