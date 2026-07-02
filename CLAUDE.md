# CLAUDE.md — mace-stream

## Project overview

Fork of [ACEsuit/mace](https://github.com/ACEsuit/mace). Active branch: `MDP-derivatives-with-bec_training`. The `DipolePolarizabilityMACE` model type (`AtomicDielectricMACE` class) computes dipole, polarizability, Born Effective Charges (BEC), and Raman tensors.

**Direct BEC training is implemented** (data pipeline + loss + training-loop wiring + CLI). Current work is **accuracy tuning**, driven by an SFG problem: IR/Raman are good but SFG is poor because the dipole derivative in the molecular-frame direction **orthogonal to the OH bond** is under-fit. Levers added for this: a relative BEC loss (`--bec_loss_eps`), per-component loss logging, and bond-frame (∥/⊥) BEC diagnostics. Raman training is still future work (a slot is left in the loss).

---

## Key files

| File | Role |
|---|---|
| `mace/modules/utils.py` | `compute_dielectric_gradients`, `compute_dielectric_gradients_loop`, `compute_bec_sparse` (placeholder) |
| `mace/modules/models.py` | `AtomicDielectricMACE.forward()` — the model |
| `mace/modules/loss.py` | Loss classes. `DipolePolarBECLoss` (`--loss dipole_polar_bec`) is the BEC-training loss; `mean_squared_error_bec` + `relative_error_bec`; `DipolePolarLoss` is the dipole+polar-only loss |
| `mace/data/atomic_data.py` | `AtomicData` — per-config data class; holds `bec` `[N,3,3]` + `bec_weight` |
| `mace/data/utils.py` | Data loading from xyz files into `AtomicData`; `bec` read from per-atom `arrays` (`--bec_key`) |
| `mace/tools/model_script_utils.py` | `configure_model()` — builds `output_args` (incl. `"bec"`) that controls what the model computes during training |
| `mace/tools/train.py` | Training loop; `bec_model_kwargs()` gates BEC derivatives; `MACELoss` metric computes `rmse_bec` + per-component loss %; `train()` has `start_bec_epoch` staging |
| `mace/tools/scripts_utils.py` | `get_loss_fn()` — selects loss based on `args.loss` string |
| `mace/cli/run_train.py` | Training entry point; sets `args.compute_dipole/polarizability/bec` for `AtomicDielectricMACE` |
| `mace/calculators/mace.py` | `MACECalculator` — inference; `get_dielectric_derivatives()` |
| `AUX/eval_all_splits.py` | Post-training eval → writes `MACE_*`/`REF_*` (incl. per-atom `bec` via `--compute_bec`) to `*.out.xyz` |
| `AUX/plot_eval.py` | Parity plots: `dipole`/`polarizability`/`energy`/`bec`; `--project-oh` adds the bond-frame ∥/⊥ decomposition |
| `tests/test_bec_training.py` | BEC data round-trip, loss (MSE/relative/masking/guard), end-to-end CLI training |
| `tests/test_calculator.py` | `test_calculator_bec_raman`, `test_calculator_bec_raman_via_get_property` |
| `tests/test_polar_models.py` | Training-path tests for DipolePolarizabilityMACE (skipped locally: `graph_longrange` not installed) |

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

### How the training loop calls the model (train.py, `take_step`/`evaluate`)

```python
output = model(
    batch_dict,
    training=True,
    compute_force=output_args["forces"],
    compute_virials=output_args["virials"],
    compute_stress=output_args["stress"],
    **bec_model_kwargs(output_args, batch, training=True),  # BEC derivative gate
)
loss = loss_fn(pred=output, ref=batch)
loss.backward()
```

`output_args` is built in `configure_model()` (model_script_utils.py) and **includes `"bec"`**:
```python
output_args = {
    "energy": ..., "forces": ..., "virials": ..., "stress": ...,
    "dipoles": args.compute_dipole,
    "polarizabilities": args.compute_polarizability,
    "bec": getattr(args, "compute_bec", False),
}
```

`bec_model_kwargs(output_args, batch, training)` (train.py) returns `{}` unless BEC is on **and** the batch actually carries labels (`(batch.bec_weight > 0).any()`), else:
```python
{"compute_dielectric_derivatives": True,
 "create_graph_for_derivatives": training,   # second-order graph only while training
 "compute_raman_tensors": False}             # no Raman training yet
```
So the expensive second-order VJP is built **only** on batches with BEC labels, and only during training (eval computes BEC with `create_graph=False`). Wired into `take_step`, `take_step_lbfgs`, and `evaluate`.

### How `AtomicDielectricMACE` is identified in run_train.py

```python
elif args.model == "AtomicDielectricMACE":
    args.compute_dipole = True
    args.compute_polarizability = True
    args.compute_energy = False
    args.compute_forces = False
    args.compute_bec = args.loss == "dipole_polar_bec"   # enables BEC training
    ...
```
`tools.train(..., start_bec_epoch=getattr(args, "start_bec_epoch", 0))` allows delayed BEC onset.

Note: the CLI arg is `--model AtomicDielectricMACE` but the calculator uses `model_type="DipolePolarizabilityMACE"`.

### Loss functions for this model (loss.py)

- `DipolePolarLoss` (`--loss dipole_polar`): dipole + polarizability MSE. Still valid for non-BEC runs.
- **`DipolePolarBECLoss` (`--loss dipole_polar_bec`)**: adds a BEC term. Key points:
  - **BEC term is guarded**: if `pred["bec"] is None` (batch had no labels → derivatives skipped) it contributes 0.
  - `component_losses(ref, pred)` returns each **weighted** term (`dipole`, `polarizability`, `bec`); `forward()` just sums it — used by the metric to report per-component loss %.
  - `bec_loss_eps` (CLI `--bec_loss_eps`, default 0) selects the BEC loss form:
    - `0` → `mean_squared_error_bec` (plain MSE, dominated by the large along-bond element).
    - `>0` → `relative_error_bec`: `w·(ref−pred)² / (ref² + eps²)`. Balances tensor elements **in any frame**, forcing the small molecular-frame ⊥-to-OH component to be fit; `eps` is a noise floor (set near the finite-difference BEC noise, ~1e-2 e) so tiny/noisy elements aren't chased. **This is the main lever for the SFG accuracy problem.**
  - `raman_weight` slot left for future Raman supervision.
- Both `mean_squared_error_bec` and `relative_error_bec` weight per-config via `bec_weight` (0 for unlabeled configs → subset training works automatically).

### Data pipeline — BEC labels (implemented)

`AtomicData` holds:
- `dipole` `[,3]` (per-graph, `REF_dipole`), `polarizability` `[1,3,3]` (per-graph, `REF_polarizability`)
- **`bec` `[N,3,3]` + scalar `bec_weight`** — per-atom, read from xyz `arrays` under `--bec_key` (e.g. `REF_bec`, stored `[N,9]` row-major → reshaped to `[N,3,3]`). Batches concatenate along the atom axis like `forces`.
- **Subset labeling is automatic**: `config_from_atoms` sets `bec_weight = 0` when the array is absent, so only labeled configs contribute (and trigger the expensive derivative).

### Training-log fields (DipolePolarRMSE error table)

`MACELoss` (train.py) additionally reports, when BEC is present:
- `rmse_bec` (printed as `RMSE_BEC=… me`, i.e. ×1000 → milli-e; RMSE over all N×3×3 labeled elements, unlabeled configs excluded via `filter_nonzero_weight`).
- `pct_loss_{dipole,polarizability,bec}` (printed as `loss%[mu/pol/bec]=…`), each = weighted term / total × 100 over the validation set. Tells you what's driving the optimizer — retune `--bec_weight` if BEC's share is too small/large. Both come from `component_losses`; absent for losses that don't expose it (back-compatible).

**Staged reporting during pretraining (`--start_bec_epoch K`).** For epochs `< K`, BEC is not trained but validation still **computes and prints `RMSE_BEC`** (so you can watch it improve indirectly as ∂μ/∂R while dipole trains) — marked `RMSE_BEC=… me (not in loss)`. In that phase the BEC term is **excluded** from the printed `loss=`, the LR scheduler, and early-stopping, and `loss%` drops to `[mu/pol]` (no bec). At epoch `K` it reverts to the full form (`RMSE_BEC=… me`, `loss%[mu/pol/bec]=…`); expect `loss=` to **jump up** at `K` as the BEC term enters. Controlled by `evaluate(..., bec_in_loss=output_args["bec"] and epoch >= start_bec_epoch)` → `MACELoss(bec_in_loss=…)`; note this is *reporting/scheduler* staging in `evaluate`, separate from the *training* staging (local `output_args` rebind in `train_one_epoch`). Defaults to `True` → non-staged/non-BEC runs unchanged.

### The critical `create_graph` constraint

During training, `loss.backward()` must differentiate through the BEC computation (itself a `torch.autograd.grad` VJP) to reach model parameters — requires `create_graph=True` in the VJP (`create_graph_for_derivatives=True`). Without it `bec` has no gradient path to parameters and contributes nothing. This builds a second-order graph over N atoms × 3 components each step (dominant cost), which is why `bec_model_kwargs` gates it to labeled batches. To manage cost:
- Delay BEC onset with `--start_bec_epoch K` (pretrain dipole+polar first), or restart-fine-tune from a dipole+polar checkpoint.
- Keep `--bec_weight` sensible (watch the `loss%` field).

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

### Batched BEC reshape (models.py forward — important for training)
`compute_dielectric_gradients` returns `[3*n_graphs, N, 3]` for a batched input, **not** `[3, N, 3]`. The correct build is reshape to `[n_graphs, C, N, 3]` and **sum over the graph axis** (cross-graph derivative blocks are structurally zero), then permute — reduces to the old single-graph `permute(1,0,2)` when `n_graphs==1`:
```python
bec = dmu_dr.view(num_graphs, 3, n_atoms, 3).sum(dim=0).permute(1, 0, 2).contiguous()          # [N,3,3]
raman = dalpha_dr.view(num_graphs, 9, n_atoms, 3).sum(dim=0).permute(1,0,2).contiguous().reshape(n_atoms,3,3,3)
```
The old `dmu_dr.permute(1,0,2)` only worked because inference uses `batch_size=1`; with `batch_size>1` it gives `[N, 3*n_graphs, 3]` and BEC-loss shape errors. Do not revert.

---

## Running BEC training

```bash
python mace/cli/run_train.py --model AtomicDielectricMACE \
    --loss dipole_polar_bec \
    --dipole_key REF_dipole --polarizability_key REF_polarizability --bec_key REF_bec \
    --dipole_weight 1000 --polarizability_weight 2000 --bec_weight 100 \
    --bec_loss_eps 1e-2 \        # >0 → relative BEC loss (the SFG lever); 0 → plain MSE
    --start_bec_epoch 0 \        # >0 delays BEC onset (pretrain dipole+polar first)
    --error_table DipolePolarRMSE --default_dtype float64 --device cuda ...
```
Example submission script: `~/NH3_NH4/NH3/dipole_and_pol_derivatives/ML/ML/models_6/submit_MDP_mace_dais.sh`.

Note: `--num_channels` sets the hidden representation width; `--MLP_irreps` is only the readout MLP. The equivariant readout (`…x1o+…x2e`) carries the off-axis (⊥) dipole derivative — widening it is a capacity lever for the SFG problem.

## Post-training evaluation & plotting (AUX/)

```bash
python AUX/eval_all_splits.py --train train.xyz --val val.xyz --test test.xyz \
    --model mace_mu_alpha_bec.model --model_type DipolePolarizabilityMACE --device cuda \
    --ref_dipole_key REF_dipole --ref_polarizability_key REF_polarizability \
    --compute_bec --ref_bec_key REF_bec          # writes per-atom MACE_bec/REF_bec to *.out.xyz

python AUX/plot_eval.py bec test.out.xyz train.out.xyz val.out.xyz            # lab-frame, 9 panels
python AUX/plot_eval.py bec test.out.xyz --project-oh                          # ALSO bond-frame 2x2
```
`--project-oh` (bec only) rotates each central atom's BEC into its bond frame and adds a **2×2 ∥/⊥ decomposition** (`∂μ∥/∂R∥`, `∂μ∥/∂R⊥`, `∂μ⊥/∂R∥`, `∂μ⊥/∂R⊥`) **in addition to** the 9-panel lab-frame figure. The `∂R⊥` / `∂μ⊥` panels are the SFG-critical directions. Bond identified by `--bond-central`/`--bond-partner`/`--bond-cutoff` (defaults H/O/1.3 Å; use `--bond-partner N` for ammonia); minimum-image applied when periodic. Output: 9-panel → `plot_bec.*` (or `--output NAME`), 2×2 → `plot_bec_HO.*` (or `NAME_oh.*`).

## Running tests

```bash
pytest tests/test_bec_training.py -v                  # BEC data/loss/end-to-end (env: mace-stream_local)
pytest tests/test_calculator.py::test_calculator_bec_raman -v
pytest tests/test_calculator.py::test_calculator_bec_raman_via_get_property -v
pytest tests/test_polar_models.py -v                  # skipped locally (graph_longrange not installed)
```

---

## Cluster paths

- Local: `/home/litmany/codes/MACE-family/mace-stream/`
- Cluster: `/dais/u/litmany/codes/mace-stream/`

Sync after local edits before running cluster jobs.

---

## i-PI socket interface (inference / MD)

`test_Yair/bulk_128_00/run-mace-mdp.py` + `socketIO_mdp.py` — drives i-PI MD. One BEC evaluation per step. Use `device="cuda"` (not `"cpu"`) on GPU machines.
