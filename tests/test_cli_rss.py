import os
import shutil
from pathlib import Path

import ase.io
import numpy as np
import pytest
from click.testing import CliRunner
try:
    from quippy.potential import Potential
    from wfl.cli.gap_rss_iter_fit import cli
except:
    pytestmark = pytest.mark.skip('quippy.potential.Potential or wfl.cli.gap_rss_iter_fit.cli was not imported')


assets_dir = Path(__file__).parent.resolve() / "assets" / "cli_rss"


def check_step(runner, step_type, seeds, iter_i):
    run_iter_s = f"run_iter_{iter_i}"
    run_iter = Path(run_iter_s)

    if "GAP_RSS_TEST_SETUP" not in os.environ:
        # copy files that cannot be created by CI (i.e. buildcell and vasp output) for step
        # from assets_dir
        run_iter.mkdir(parents=True, exist_ok=True)
        for fn in (assets_dir / run_iter_s).glob("initial_random_configs.*.xyz"):
            shutil.copy(fn, run_iter / fn.name)
        for fn in (assets_dir / run_iter_s).glob("DFT_evaluated_*.xyz"):
            shutil.copy(fn, run_iter / fn.name)

    # actually run step
    result = runner.invoke(cli, ["-c", "LiCu.json", "--seeds", seeds, step_type])
    # make sure it ran and created something
    assert result.exit_code == 0
    assert (run_iter / f"GAP_iter_{iter_i}.xml").exists()

    if "GAP_RSS_TEST_SETUP" in os.environ:
        # save this iter's hard-to-generate (output of buildcell, vasp) files in assets_dir
        (assets_dir / run_iter_s).mkdir(parents=True, exist_ok=True)
        for fn in run_iter.glob("initial_random_configs.*.xyz"):
            shutil.copy(fn, assets_dir / run_iter_s / fn.name)
        for fn in run_iter.glob("DFT_evaluated_*.xyz"):
            shutil.copy(fn, assets_dir / run_iter_s / fn.name)

    # save or use results so we can test that they are correct by checking GAP predictions
    if "GAP_RSS_TEST_SETUP" in os.environ:
        # save energies of some configs for actual test
        ats = ase.io.read(
            run_iter / f"testing.error_database.GAP_iter_{iter_i}.xyz", ":"
        )
        pot = Potential(param_filename=str(run_iter / f"GAP_iter_{iter_i}.xml"))
        with open(assets_dir / run_iter_s / "cli_rss_test_energies", "w") as fout:
            for at in ats:
                at.calc = pot
                fout.write(f"{at.get_potential_energy()}\n")
    else:
        # test by comparing to saved energies
        ats = ase.io.read(
            run_iter / f"testing.error_database.GAP_iter_{iter_i}.xyz", ":"
        )
        pot = Potential(param_filename=str(run_iter / f"GAP_iter_{iter_i}.xml"))
        with open(assets_dir / run_iter_s / "cli_rss_test_energies") as fin:
            ref_energies = [float(l.strip()) for l in fin.readlines()]
        for at, energy in zip(ats, ref_energies):
            at.calc = pot
            assert np.abs(at.get_potential_energy() - energy) < 1.0e-5


def do_full_test(runner, assets_dir):
    # copy in config files (for prep)
    for fn in [
        "LiCu.json",
        "length_scales.yaml",
        "multistage_GAP_fit_settings.template.yaml",
    ]:
        shutil.copy(assets_dir / fn, fn)

    # make sure env vars and basic files for buildcell, vasp are ready
    if "GAP_RSS_TEST_SETUP" in os.environ:
        # real VASP, make sure it is configured
        assert "VASP_COMMAND" in os.environ
        assert "VASP_COMMAND_GAMMA" in os.environ
        assert "VASP_PP_PATH" in os.environ
        # use real buildcell
        assert "GRIF_BUILDCELL_CMD" in os.environ
    else:
        # fake VASP, create dummy_potcars for ase calculators and set same path in VASP_PP_PATH
        (Path("dummy_potcars") / "Li").mkdir(parents=True, exist_ok=True)
        (Path("dummy_potcars") / "Cu").mkdir(parents=True, exist_ok=True)
        with open(Path("dummy_potcars") / "Li" / "POTCAR", "w") as fout:
            fout.write("\n")
        with open(Path("dummy_potcars") / "Cu" / "POTCAR", "w") as fout:
            fout.write("\n")
        os.environ["VASP_PP_PATH"] = "dummy_potcars"

    # make sure nothing is parallel so things are as deterministic as possible
    os.environ["WFL_DETERMINISTIC_HACK"] = "1"
    os.environ["WFL_AUTOPARA_NPOOL"] = "0"

    # PREP
    result = runner.invoke(cli, ["-c", "LiCu.json", "--seeds", "42,43,44", "prep"])
    assert result.exit_code == 0

    # ACTUAL STEPS
    check_step(runner, "initial_step", "45,46,47", "0")
    check_step(runner, "rss_step", "48,49,50", "1")
    check_step(runner, "MD_bulk_defect_step", "51,52,52", "2")


# @pytest.mark.skip(reason="too computationally expensive")
@pytest.mark.slow
def test_cli_rss_full(tmp_path):
    runner = CliRunner()
    if "GAP_RSS_TEST_SETUP" in os.environ:
        # setup run that actually does work and creates files
        orig_dir = os.getcwd()
        try:
            os.chdir(os.environ.get("GAP_RSS_TEST_SETUP"))
            do_full_test(runner, assets_dir)
        finally:
            os.chdir(orig_dir)
    else:
        with runner.isolated_filesystem(tmp_path):
            do_full_test(runner, assets_dir)
