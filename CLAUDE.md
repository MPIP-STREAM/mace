# CLAUDE.md — mace-stream

## Project overview

Fork of [ACEsuit/mace](https://github.com/ACEsuit/mace). Active branch: `MDP-derivatives-with-bec_training`. The `DipolePolarizabilityMACE` model type (`AtomicDielectricMACE` class) computes dipole, polarizability, Born Effective Charges (BEC), and Raman tensors. Current work: add BEC (and possibly Raman) training support, i.e. supervise the model on BEC labels.

---

## Key files

| File | Role |
|---|---|
| `mace/modules/utils.py` | `compute_dielectric_gradients`, `compute_dielectric_gradients_loop`, `compute_bec_sparse` (placeholder) |
| `mace/modules/models.py` | `AtomicDielectricMACE.forward()` — the model |
| `mace/modules/loss.py` | All loss classes; `DipolePolarLoss` is current loss for this model |
| `mace/data/atomic_data.py` | `AtomicData` — per-config data class; defines what labels exist in the pipeline |
| `mace/data/utils.py` | Data loading from xyz files into `AtomicData` |
| `mace/tools/model_script_utils.py` | `configure_model()` — builds `output_args` dict that controls what the model computes during training |
| `mace/tools/train.py` | `take_step()`, `train_one_epoch()` — training loop; calls `model(batch_dict, training=True, **output_args)` then `loss.backward()` |
| `mace/tools/scripts_utils.py` | `get_loss_fn()` — selects loss based on `args.loss` string |
| `mace/cli/run_train.py` | Training entry point; sets `args.compute_dipole`, `args.compute_polarizability`, etc. for `AtomicDielectricMACE` |
| `mace/calculators/mace.py` | `MACECalculator` — inference; `get_dielectric_derivatives()` |
| `tests/test_calculator.py` | `test_calculator_bec_raman`, `test_calculator_bec_raman_via_get_property` |
| `tests/test_polar_models.py` | Training-path tests for DipolePolarizabilityMACE |

---

## What BEC/Raman computation does

- **BEC** (`bec` key in model output): `d(μ_α)/d(R_Iβ)`, shape `[N_atoms, 3, 3]` — 3 VJPs over dipole components
- **Raman** (`raman_tensors` key): `d(α_αβ)/d(R_Iγ)`, shape `[N_atoms, 3, 3, 3]` — 9 VJPs over polarizability components
- Also available (raw, unpermuted): `dmu_dr` `[3, N, 3]`, `dalpha_dr` `[9, N, 3]`
- Computed in `compute_dielectric_gradients` → `compute_dielectric_gradients_loop` (on CUDA)

---

## `AtomicDielectricMACE.forward()` — key parameters (models.py ~line 1020)

```python
def forward(
    data: Dict[str, torch.Tensor],
    training: bool = False,
    compute_dielectric_derivatives: bool = False,  # gate — skip all VJPs unless True
    compute_raman_tensors: bool = True,             # skip 9 Raman VJPs when False
    create_graph_for_derivatives: bool = False,     # MUST be True when training on BEC/Raman
    ...
)
```

- `compute_dielectric_derivatives=False` → `bec`, `raman_tensors`, `dmu_dr`, `dalpha_dr` are all `None` in output.
- `create_graph_for_derivatives=False` (default) → inference mode. The VJP does NOT build a second-order graph, so `loss.backward()` cannot propagate through BEC back to model parameters. Fast, no memory growth.
- `create_graph_for_derivatives=True` → training mode. Builds the second-order graph during each VJP so `loss.backward()` can differentiate through BEC. **Required for training on BEC labels.** Expensive: ~30× overhead vs inference was observed before other fixes; the remaining cost of the second-order graph during training is unavoidable.

---

## Training infrastructure

### How the training loop calls the model (train.py ~line 418)

```python
output = model(
    batch_dict,
    training=True,
    compute_force=output_args["forces"],
    compute_virials=output_args["virials"],
    compute_stress=output_args["stress"],
)
loss = loss_fn(pred=output, ref=batch)
loss.backward()
```

`output_args` is built in `configure_model()` (model_script_utils.py ~line 42):
```python
output_args = {
    "energy": args.compute_energy,
    "forces": args.compute_forces,
    "virials": compute_virials,
    "stress": compute_stress,
    "dipoles": args.compute_dipole,
    "polarizabilities": args.compute_polarizability,
}
```

Currently there is **no `bec` or `raman` in `output_args`**, and `compute_dielectric_derivatives` / `create_graph_for_derivatives` are never passed during training. This is what needs to be added.

### How `AtomicDielectricMACE` is identified in run_train.py (~line 545)

```python
elif args.model == "AtomicDielectricMACE":
    atomic_energies = None
    dipole_only = False
    args.compute_dipole = True
    args.compute_polarizability = True
    args.compute_energy = False
    args.compute_forces = False
    ...
```

Note: the CLI arg is `--model AtomicDielectricMACE` but the calculator uses `model_type="DipolePolarizabilityMACE"`.

### Existing loss functions (loss.py)

The current loss for this model type is `DipolePolarLoss` (selected by `--loss dipole_polar`):
```python
class DipolePolarLoss(torch.nn.Module):
    def forward(self, ref: Batch, pred: TensorDict, ...) -> torch.Tensor:
        loss_dipole = weighted_mean_squared_error_dipole(ref, pred)          # MSE on dipole [n_graphs, 3]
        loss_polarizability = weighted_mean_squared_error_polarizability(ref, pred)  # MSE on [n_graphs, 3, 3]
        return dipole_weight * loss_dipole + polarizability_weight * loss_polarizability
```

The pattern for a BEC loss would follow `mean_squared_error_forces` (forces are also per-atom, shape `[N, 3]`):
```python
def mean_squared_error_forces(ref, pred, ddp=None):
    configs_weight = torch.repeat_interleave(ref.weight, ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(ref.forces_weight, ...).unsqueeze(-1)
    raw_loss = configs_weight * configs_forces_weight * torch.square(ref["forces"] - pred["forces"])
    return reduce_loss(raw_loss, ddp)
```

BEC has shape `[N, 3, 3]` — similar per-atom structure but 3×3 instead of 3.

### Data pipeline — what labels exist today (atomic_data.py)

`AtomicData.__init__` currently accepts (relevant subset):
- `dipole` `[, 3]` — per-graph total dipole (label key: `REF_dipole`)
- `polarizability` `[1, 3, 3]` — per-graph polarizability (label key: `REF_polarizability`)
- `forces` `[N, 3]` — per-atom forces

**BEC labels do not yet exist in the data pipeline.** Adding them requires:
1. A new field `bec: Optional[torch.Tensor]  # [N, 3, 3]` and `bec_weight` in `AtomicData`
2. Data loading in `data/utils.py` to read BEC from xyz `arrays` (per-atom, not `info`)
3. The batching machinery handles per-atom tensors automatically (same as forces)

---

## What needs to be implemented for BEC training

### Minimum viable path

1. **Data pipeline** (`mace/data/atomic_data.py`, `mace/data/utils.py`):
   - Add `bec: Optional[torch.Tensor]  # [N_atoms, 3, 3]` field to `AtomicData`
   - Add `bec_weight` per-config scalar weight (follow `forces_weight` pattern)
   - Load BEC from xyz `arrays` key (e.g. `REF_bec`, shape `[N, 3, 3]` per atom)

2. **Loss function** (`mace/modules/loss.py`):
   - Add `mean_squared_error_bec(ref, pred, ddp)` — per-atom MSE on `[N, 3, 3]` tensor
   - Add `DipolePolarBECLoss` (or extend `DipolePolarLoss`) with `bec_weight` term
   - Register new loss name in `get_loss_fn()` in `scripts_utils.py`

3. **Training loop** (`mace/tools/train.py`, `mace/tools/model_script_utils.py`):
   - Add `"bec": args.compute_bec` to `output_args` in `configure_model()`
   - In `take_step()` / the distributed training closure: when `output_args["bec"]`, pass `compute_dielectric_derivatives=True, create_graph_for_derivatives=True` to the model call
   - Note: `compute_raman_tensors` should be False unless training on Raman too (avoid 9 extra VJPs)

4. **CLI args** (`mace/tools/arg_parser.py` or `arg_parser_tools.py`):
   - Add `--bec_weight` argument
   - Add `--loss dipole_polar_bec` option

5. **`run_train.py`**:
   - In the `AtomicDielectricMACE` branch, set `args.compute_bec = True` when `args.loss` involves BEC

### The critical `create_graph` constraint

During training, `loss.backward()` must differentiate through the BEC computation (which is itself a `torch.autograd.grad` call) to reach model parameters. This requires `create_graph=True` in the VJP, set via `create_graph_for_derivatives=True` on `forward()`. Without it, `bec` in the model output has no gradient path to model parameters and contributes nothing to training.

This makes each training step expensive (second-order graph over N atoms × 3 components). Consider:
- Only enabling BEC loss after initial dipole+polarizability pretraining converges (two-stage training)
- Using a small `bec_weight` relative to dipole/polarizability losses
- Gradient checkpointing if memory is an issue

---

## Inference — current state (working)

```python
calc = MACECalculator(
    model_paths="MACE-MDP.model",
    device="cuda",
    model_type="DipolePolarizabilityMACE",
)
calc.calculate(atoms, properties=["bec"])            # 3 VJPs, create_graph=False
calc.calculate(atoms, properties=["bec", "raman_tensors"])  # 12 VJPs, create_graph=False
```

`compute_raman_tensors` is set automatically from `properties`. `create_graph_for_derivatives` defaults to `False` in inference — do not change this.

---

## Known implementation details / gotchas

### `retain_graph_after` (utils.py ~line 537)
When BEC and Raman are both requested, they share the same forward graph. BEC's last VJP must NOT free the graph. Fixed by passing `retain_graph_after=compute_raman_tensors` to the BEC call in `forward()`. Training with BEC+Raman will also need this.

### CUDA path (utils.py ~line 557)
`torch.func.vmap + torch.autograd.grad` crashes on CUDA. The function bypasses vmap on GPU:
```python
if positions.is_cuda:
    return compute_dielectric_gradients_loop(dielectric, positions, ...)
```
The loop handles `retain_graph` correctly.

### `compute_bec_sparse` (utils.py ~line 613)
Placeholder for a future sparse BEC implementation using MACE locality (nonzero `d(μ_j)/d(R_I)` only within `r_cut`). Not implemented. Do not use.

### `get_dielectric_derivatives()` (mace.py ~line 756)
Inference-only convenience method. Returns `bec` and `raman_tensors` (already permuted/shaped), and stores them in `self.results`. Separate from `calculate()`.

---

## Running tests

```bash
pytest tests/test_calculator.py::test_calculator_bec_raman -v
pytest tests/test_calculator.py::test_calculator_bec_raman_via_get_property -v
pytest tests/test_polar_models.py -v
```

---

## Cluster paths

- Local: `/home/litmany/codes/MACE-family/mace-stream/`
- Cluster: `/dais/u/litmany/codes/mace-stream/`

Sync after local edits before running cluster jobs.

---

## i-PI socket interface (inference / MD)

`test_Yair/bulk_128_00/run-mace-mdp.py` + `socketIO_mdp.py` — drives i-PI MD. One BEC evaluation per step. Use `device="cuda"` (not `"cpu"`) on GPU machines.
