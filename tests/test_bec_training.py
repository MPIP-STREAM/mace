"""Tests for direct Born-effective-charge (BEC) supervision.

Covers the new data-pipeline plumbing, the BEC loss term, and an end-to-end
training run that exercises the second-order (create_graph) gradient path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from ase import Atoms

from mace import data
from mace.calculators import MACECalculator
from mace.modules.loss import (
    DipolePolarBECLoss,
    mean_squared_error_bec,
    relative_error_bec,
)
from mace.tools import torch_geometric, utils

run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"

Z_TABLE = utils.AtomicNumberTable([1, 8])
CUTOFF = 4.0


def _water(with_bec: bool, seed: int = 0) -> Atoms:
    rng = np.random.default_rng(seed)
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.927, 0.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )
    atoms.positions += rng.normal(0, 0.05, size=atoms.positions.shape)
    # ASE drops a plain info["dipole"] on extxyz round-trip (reserved key), so
    # use REF_ keys for the file-backed training path.
    atoms.info["REF_dipole"] = rng.normal(0, 0.1, size=3)
    atoms.info["REF_polarizability"] = rng.normal(0, 0.1, size=(3, 3))
    if with_bec:
        # per-atom 9-column row-major flatten of [3, 3]
        atoms.new_array("bec", rng.normal(0, 0.1, size=(len(atoms), 9)))
    return atoms


def _atomic_data(atoms: Atoms) -> data.AtomicData:
    keyspec = data.KeySpecification.from_defaults()
    config = data.config_from_atoms(atoms, key_specification=keyspec)
    return data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=CUTOFF)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def test_bec_roundtrip_labeled():
    arr = np.arange(3 * 9, dtype=float).reshape(3, 9)
    atoms = _water(with_bec=False)
    atoms.new_array("bec", arr)
    ad = _atomic_data(atoms)

    assert ad.bec.shape == (3, 3, 3)
    assert float(ad.bec_weight) == 1.0
    # atom 0 row 0..8 reshapes row-major to [[0,1,2],[3,4,5],[6,7,8]]
    np.testing.assert_allclose(ad.bec[0].numpy(), arr[0].reshape(3, 3))


def test_bec_absent_gives_zero_weight():
    ad = _atomic_data(_water(with_bec=False))
    assert ad.bec.shape == (3, 3, 3)
    assert float(ad.bec_weight) == 0.0
    assert torch.count_nonzero(ad.bec) == 0


def test_bec_batches_along_atom_axis():
    labeled = _atomic_data(_water(with_bec=True, seed=1))
    unlabeled = _atomic_data(_water(with_bec=False, seed=2))
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[labeled, unlabeled], batch_size=2, shuffle=False
    )
    batch = next(iter(loader))
    assert batch.bec.shape == (6, 3, 3)  # 3 + 3 atoms
    np.testing.assert_allclose(batch.bec_weight.numpy(), [1.0, 0.0])


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _batch(*atoms_list):
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[_atomic_data(a) for a in atoms_list],
        batch_size=len(atoms_list),
        shuffle=False,
    )
    return next(iter(loader))


def test_mean_squared_error_bec_zero_when_exact():
    ref = _batch(_water(with_bec=True, seed=3))
    pred = {"bec": ref["bec"].clone()}
    assert float(mean_squared_error_bec(ref, pred)) == pytest.approx(0.0)


def test_mean_squared_error_bec_masks_unlabeled():
    # One labeled + one unlabeled config; a large error on the unlabeled atoms
    # must not contribute because its bec_weight is 0.
    ref = _batch(_water(with_bec=True, seed=4), _water(with_bec=False, seed=5))
    pred = {"bec": ref["bec"].clone()}
    pred["bec"][3:] += 100.0  # corrupt the unlabeled config's atoms
    assert float(mean_squared_error_bec(ref, pred)) == pytest.approx(0.0)


def test_dipole_polar_bec_loss_guards_missing_bec():
    ref = _batch(_water(with_bec=True, seed=6))
    pred = {
        "dipole": ref["dipole"].clone(),
        "polarizability": ref["polarizability"].view(-1, 3, 3).clone(),
    }
    loss_fn = DipolePolarBECLoss(
        dipole_weight=1.0, polarizability_weight=1.0, bec_weight=1.0
    )
    # pred has no "bec" -> BEC term skipped, dipole+polar are exact -> 0
    assert float(loss_fn(ref, pred)) == pytest.approx(0.0)

    pred["bec"] = ref["bec"].clone() + 1.0
    assert float(loss_fn(ref, pred)) > 0.0


def test_relative_error_bec_zero_when_exact():
    ref = _batch(_water(with_bec=True, seed=7))
    pred = {"bec": ref["bec"].clone()}
    assert float(relative_error_bec(ref, pred, eps=1e-2)) == pytest.approx(0.0)


def test_relative_error_bec_upweights_small_components():
    # A fixed absolute error on a SMALL reference element must count for more,
    # under the relative loss, than the SAME absolute error on a LARGE element.
    ref = _batch(_water(with_bec=False, seed=8))  # weight applied manually below
    ref.bec_weight = torch.ones_like(ref.bec_weight)
    big = ref["bec"].clone()
    small = ref["bec"].clone()
    big[0, 0, 0] = 10.0
    small[0, 0, 0] = 0.1
    delta = 0.05

    # same absolute error delta on the [0,0,0] element in each case
    pred_big = {"bec": big.clone()}
    pred_big["bec"][0, 0, 0] += delta
    pred_small = {"bec": small.clone()}
    pred_small["bec"][0, 0, 0] += delta

    ref.bec = big
    loss_big = float(relative_error_bec(ref, pred_big, eps=1e-3))
    ref.bec = small
    loss_small = float(relative_error_bec(ref, pred_small, eps=1e-3))
    # small-reference element contributes a much larger relative loss
    assert loss_small > loss_big


def test_relative_error_bec_eps_caps_noise():
    # For |ref| << eps the denominator ~ eps^2, so the relative loss reduces to
    # a scaled MSE and does not blow up as ref -> 0.
    ref = _batch(_water(with_bec=False, seed=9))
    ref.bec_weight = torch.ones_like(ref.bec_weight)
    ref.bec = torch.zeros_like(ref.bec)
    pred = {"bec": torch.full_like(ref.bec, 0.01)}
    eps = 1.0
    got = float(relative_error_bec(ref, pred, eps=eps))
    # every element error^2 / eps^2 = 1e-4, averaged -> 1e-4
    assert got == pytest.approx(1e-4, rel=1e-6)


# ---------------------------------------------------------------------------
# End-to-end training (exercises create_graph second-order path)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", name="bec_train_dir")
def bec_train_dir_fixture(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("bec_run_")
    isolated = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3),
    ]
    for a, e in zip(isolated, (1.0, -0.5)):
        a.info["REF_energy"] = e
        a.info["config_type"] = "IsolatedAtom"
    # Subset labeling: only ~half the configs carry BEC.
    configs = list(isolated)
    for i in range(16):
        configs.append(_water(with_bec=(i % 2 == 0), seed=100 + i))
    ase.io.write(tmp_path / "fit.xyz", configs)

    params = {
        "name": "MACE",
        "valid_fraction": 0.1,
        "model": "AtomicDielectricMACE",
        "num_channels": 8,
        "max_L": 1,
        "r_max": 3.5,
        "batch_size": 4,
        "max_num_epochs": 3,
        "MLP_irreps": "16x0e+16x1o+16x2e",
        "device": "cpu",
        "seed": 5,
        "loss": "dipole_polar_bec",
        "bec_weight": 1.0,
        "dipole_weight": 1.0,
        "polarizability_weight": 1.0,
        "energy_key": "",
        "forces_key": "",
        "stress_key": "",
        "dipole_key": "REF_dipole",
        "polarizability_key": "REF_polarizability",
        "bec_key": "bec",
        "error_table": "DipolePolarRMSE",
        "eval_interval": 1,
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "train_file": str(tmp_path / "fit.xyz"),
    }
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    cmd = [sys.executable, str(run_train)] + [
        (f"--{k}={v}" if v is not None else f"--{k}") for k, v in params.items()
    ]
    proc = subprocess.run(cmd, env=run_env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
    assert proc.returncode == 0
    return tmp_path


def test_bec_training_produces_finite_bec(bec_train_dir):
    calc = MACECalculator(
        bec_train_dir / "MACE.model",
        device="cpu",
        model_type="DipolePolarizabilityMACE",
    )
    atoms = _water(with_bec=False, seed=999)
    calc.calculate(atoms, properties=["bec"])
    bec = np.asarray(calc.results["bec"])
    assert bec.shape == (3, 3, 3)
    assert np.all(np.isfinite(bec))
