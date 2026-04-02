import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms

from mace.calculators import MACECalculator

try:
    import cuequivariance as cue  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

from scipy.spatial.transform import Rotation as Rscipy

run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"


@pytest.fixture(name="fitting_configs")
def fixture_fitting_configs():
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    fit_configs = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3),
    ]
    fit_configs[0].info["REF_energy"] = 0.0
    fit_configs[0].info["config_type"] = "IsolatedAtom"
    fit_configs[1].info["REF_energy"] = 0.0
    fit_configs[1].info["config_type"] = "IsolatedAtom"

    np.random.seed(5)
    for _ in range(20):
        c = water.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        c.info["REF_energy"] = np.random.normal(0.1)
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        c.info["REF_stress"] = np.random.normal(0.1, size=6)
        c.info["REF_dipoles"] = np.random.normal(0.1, size=(3,))
        c.info["REF_polarizability"] = np.random.normal(0.1, size=(3, 3))
        fit_configs.append(c)

    return fit_configs


@pytest.fixture(name="pretraining_configs")
def fixture_pretraining_configs():
    configs = []
    for _ in range(10):
        atoms = Atoms(
            numbers=[8, 1, 1],
            positions=np.random.rand(3, 3) * 3,
            cell=[5, 5, 5],
            pbc=[True] * 3,
        )
        atoms.info["REF_energy"] = np.random.normal(0, 1)
        atoms.arrays["REF_forces"] = np.random.normal(0, 1, size=(3, 3))
        atoms.info["REF_stress"] = np.random.normal(0, 1, size=6)
        atoms.info["REF_dipoles"] = np.random.normal(0.1, size=(3,))
        atoms.info["REF_polarizability"] = np.random.normal(0.1, size=(3, 3))
        configs.append(atoms)

    configs.append(
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3),
    )
    configs.append(
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3)
    )
    configs[-2].info["REF_energy"] = -2.0
    configs[-2].info["config_type"] = "IsolatedAtom"
    configs[-1].info["REF_energy"] = -4.0
    configs[-1].info["config_type"] = "IsolatedAtom"
    return configs


_mace_params_dipole_polar_cueq = {
    "name": "DielectricMACE",
    "valid_fraction": 0.05,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "stress_weight": 1.0,
    "dipole_weight": 1.0,
    "polarizability_weight": 1.0,
    "model": "AtomicDielectricMACE",
    "r_max": 3.5,
    "max_L": 2,
    "batch_size": 5,
    "max_num_epochs": 10,
    "ema_decay": 0.99,
    "amsgrad": None,
    "restart_latest": None,
    "device": "cpu",
    "enable_cueq": True,
    "seed": 5,
    "loss": "dipole_polar",
    "MLP_irreps": "16x0e+16x1o+16x2e",
    "error_table": "DipolePolarRMSE",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "dipole_key": "REF_dipoles",
    "polarizability_key": "REF_polarizability",
    "eval_interval": 2,
    "use_reduced_cg": False,
    "compute_polarizability": True,
}


def rotate_atoms(at: Atoms, rot_mat: np.ndarray) -> Atoms:
    """Rotate positions and cell by rot_mat (3x3)."""
    at_r = at.copy()

    # positions (N,3): x' = R x  ==> using row-vectors -> pos' = pos @ R^T
    pos = at_r.get_positions()
    at_r.set_positions(pos @ rot_mat.T)

    # cell (3,3) with row cell vectors -> cell' = cell @ R^T
    cell = np.array(at_r.get_cell())
    if cell.shape == (3, 3):
        at_r.set_cell(cell @ rot_mat.T, scale_atoms=False)

    at_r.set_pbc(at.get_pbc())
    return at_r


def assert_equivariance_for_calculator(
    calc_obj,
    configs,
    configs_rot,
    Rmat,
    atol=1e-8,
    rtol=1e-8,
):
    """Check mu(rot(x)) ≈ R mu(x) and alpha(rot(x)) ≈ R alpha(x) R^T."""
    for at, at_r in zip(configs, configs_rot):
        # original
        at.calc = calc_obj
        mu = np.asarray(at.get_dipole_moment())
        alpha = np.asarray(calc_obj.get_property("polarizability", at))

        # rotated
        at_r.calc = calc_obj
        mu_r = np.asarray(at_r.get_dipole_moment())
        alpha_r = np.asarray(calc_obj.get_property("polarizability", at_r))
        print("Mus R:",mu_r, Rmat @ mu)
        print("Alphas R:",alpha_r, Rmat @ alpha @ Rmat.T)
        assert np.allclose(Rmat @ mu, mu_r, atol=atol, rtol=rtol)
        assert np.allclose(Rmat @ alpha @ Rmat.T, alpha_r, atol=atol, rtol=rtol)


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_run_train_dipole_polar_cueq_with_rotation(tmp_path, fitting_configs):
    # ---------- write training data ----------
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params_dipole_polar_cueq.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    # ensure run_train.py uses the repo under test
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)

    cmd = (
        sys.executable
        + " "
        + str(run_train)
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), env=run_env, check=True)
    assert p.returncode == 0

    model_path = tmp_path / "DielectricMACE.model"

    # ---------- calculators ----------
    calc = MACECalculator(
        model_paths=model_path,
        model_type="DipolePolarizabilityMACE",
        device="cpu",
    )
    calc_cueq = MACECalculator(
        model_paths=model_path,
        model_type="DipolePolarizabilityMACE",
        device="cpu",
        enable_cueq=True,
    )

    # ---------- build rotated configs ----------
    rot_mat = Rscipy.from_euler("z", 60, degrees=True).as_matrix()
    fitting_configs_rot = [rotate_atoms(at, rot_mat) for at in fitting_configs]

    # ---------- (optional) existing regression arrays ----------
    # If you still want the old "exact value" regression checks, keep your
    # ref_Mus/ref_alphas and do those asserts here, unchanged.
    #
    # Mus = []
    # alphas = []
    # for at in fitting_configs:
    #     at.calc = calc
    #     Mus.append(at.get_dipole_moment())
    #     alphas.append(calc.get_property("polarizability", at))
    #
    # Mus_cueq = []
    # alphas_cueq = []
    # for at in fitting_configs:
    #     at.calc = calc_cueq
    #     Mus_cueq.append(at.get_dipole_moment())
    #     alphas_cueq.append(calc_cueq.get_property("polarizability", at))
    #
    # assert np.allclose(Mus, ref_Mus)
    # assert np.allclose(alphas, ref_alphas)
    # assert np.allclose(Mus_cueq, ref_Mus)
    # assert np.allclose(alphas_cueq, ref_alphas)

    # ---------- new: equivariance checks (original vs rotated) ----------
    print("Checking cueq=False!")
    assert_equivariance_for_calculator(calc, fitting_configs, fitting_configs_rot, rot_mat)
    print("\nChecking cueq=True!\n")
    assert_equivariance_for_calculator(calc_cueq, fitting_configs, fitting_configs_rot, rot_mat)

