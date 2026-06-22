# CLAUDE.md — mace-stream

## Project overview

This is a fork of [ACEsuit/mace](https://github.com/ACEsuit/mace), a PyTorch library for training and running MACE machine-learning interatomic potentials. The active branch is `MDP-derivatives`, which adds efficient Born Effective Charge (BEC) and Raman tensor computation to the `DipolePolarizabilityMACE` model type.

## Key files for BEC/Raman work

| File | Role |
|---|---|
| `mace/modules/utils.py` | `compute_dielectric_gradients`, `compute_dielectric_gradients_loop`, `compute_bec_sparse` (placeholder) |
| `mace/modules/models.py` | `AtomicDielectricMACE.forward()` — computes BEC and (optionally) Raman tensors |
| `mace/calculators/mace.py` | `MACECalculator.calculate()` — ASE interface; sets `compute_raman_tensors` flag |
| `tests/test_calculator.py` | `test_calculator_bec_raman`, `test_calculator_bec_raman_via_get_property` |
| `aux/` | Diagnostic outputs from i-PI / job runs (read-only reference) |

## BEC/Raman design decisions

### How BEC is computed
BEC = Jacobian of dipole w.r.t. atomic positions: `d(μ_α)/d(R_Iβ)`, shape `[N_atoms, 3, 3]`.
Raman = Jacobian of polarizability w.r.t. positions: `d(α_αβ)/d(R_Iγ)`, shape `[N_atoms, 3, 3, 3]`.
Both are computed via reverse-mode AD (VJPs): 3 backward passes for BEC, 9 for Raman.

### `create_graph` — inference vs training
- **`create_graph=False` (default)**: inference mode. Does NOT build the second-order computational graph during backward. Significantly faster and avoids memory accumulation over many frames.
- **`create_graph=True`**: only needed when training on BEC/Raman labels (higher-order gradients required for loss). Enabling this at inference caused a 30x slowdown and a memory leak (timing grew 7s→12s over 50 frames).
- Exposed as `create_graph_for_derivatives` on `AtomicDielectricMACE.forward()`.

### Skip Raman when not requested
`compute_raman_tensors=False` on `forward()` skips the 9 extra VJPs entirely. The `MACECalculator` sets this automatically based on whether `"raman_tensors"` appears in the `properties` list passed to `calculate()`.

### GPU vs CPU paths in `compute_dielectric_gradients`
`torch.func.vmap` + `torch.autograd.grad` is **incompatible with CUDA tensors** (TorchScript storage access error). The function detects this at runtime:
- **CUDA**: goes directly to `compute_dielectric_gradients_loop` (serial VJPs, correct `retain_graph` handling).
- **CPU**: tries `torch.func.vmap` first (may be faster on some configs), falls back to the loop on `RuntimeError`.

### `retain_graph` in the vmap path
`torch.func.vmap` calls `torch.autograd.grad` *sequentially* (not truly batched). The forward graph must survive between calls, so `retain_graph=True` is set inside the `get_vjp` closure. The graph is freed naturally when `forward()` returns and releases `total_dipole`.

In the loop path, `retain_graph=(i < n_out - 1) or create_graph` frees the graph on the last VJP, avoiding the memory overhead.

### `compute_bec_sparse` (placeholder)
`mace/modules/utils.py` contains a stub `compute_bec_sparse` for a future per-atom reverse-neighbor-list implementation. Because MACE is local, `d(μ_j)/d(R_I)` is nonzero only when atom I is within `r_cut` of atom j. A sparse implementation would do one backward per (atom, output-component) pair using only the local subgraph, targeting O(Z) cost vs the current O(N) for large systems.

## How to use

```python
from mace.calculators import MACECalculator

# BEC only (fast — 3 VJPs, no Raman)
calc = MACECalculator(
    model_paths="MACE-MDP.model",
    device="cuda",                      # always use cuda on GPU machines
    model_type="DipolePolarizabilityMACE",
)
calc.calculate(atoms, properties=["bec"])

# BEC + Raman (12 VJPs total)
calc.calculate(atoms, properties=["bec", "raman_tensors"])
```

Do **not** use `device="cpu"` on GPU machines — this was a past mistake that made BEC take 6s/frame instead of ~1s.

## Running tests

```bash
pytest tests/test_calculator.py::test_calculator_bec_raman -v
pytest tests/test_calculator.py::test_calculator_bec_raman_via_get_property -v
pytest tests/test_polar_models.py -v
```

## Cluster paths

- Local: `/home/litmany/codes/MACE-family/mace-stream/`
- Cluster: `/dais/u/litmany/codes/mace-stream/`

After editing locally, sync to cluster before running jobs.

## Branch

All BEC/Raman performance work is on branch `MDP-derivatives`. The `main` branch is the upstream merge base.
