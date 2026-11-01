# SBS reproducibility

**Site-Balance-Structured Learning of Catalytic Kinetics From Reactor Observations**

This compact repository replays the archived CPU results for synthetic reverse
water-gas shift (RWGS), Ni/CeO2 methanation, and Cu/ZnO/Al2O3 methanol synthesis.
It retains explicit apparent rate laws, thermodynamic coupling, surface
closures, reactor models, physical parameter mappings and bounded estimation.
The methanol synthesis closures are proposed effective approximations, not exact reductions
of the source microkinetic mechanism. Predictive agreement, numerical
qualification and parameter identifiability are distinct claims.

## Install and reproduce

The verification environment is Linux x86-64, Python **3.11.16**, NumPy
**2.4.6**, SciPy **1.17.1**, and PyYAML **6.0.3**, with double precision.
No GPU, JAX, external download, or access to the original workspace is needed.

```bash
python -m pip install -r requirements.txt
python scripts/reproduce.py --case all
```

Individual cases use `--case rwgs`, `--case ni`, or `--case methanol`.
Paths are resolved relative to the script, not the current working directory.
The default destination is `outputs/reproduction/`. An existing destination
is not overwritten; for a subsequent run use `--output outputs/replay_2`.
All generated files belong outside version control and the release archive.

The default command **does not optimize, select new representatives, or search
structures**. It recomputes reactor predictions from frozen physical parameters,
errors, Figures 2-4 numerical source tables, common-state factors/orders, and
SVDs of saved data-residual Jacobians. References are used for comparisons,
not substituted for calculations. Historical task states are exported with
an `archived_` prefix and are not reassessed by a prediction replay.

## Inputs and evidence

| Case | Fixed data and partitions | Archived comparison |
|---|---|---|
| RWGS | 3 fitting runs (conditions 0/17/39); separate 40 fitting / 10 model-selection validation runs; no test set | 16 supports x 5 starts; 7 qualified representatives; 9 optimization-unresolved candidates retained |
| Ni/CeO2 | 124 records: 96 fitting / 28 model-selection validation; no test set | 18 tasks: A/B, isothermal/thermal/free-order stages, ordinary and literature-reference initializations; 3 completed conditional fixed-parameter refits |
| Methanol synthesis over Cu/ZnO/Al₂O₃ | 359 records: 227 fitting / 66 model-selection validation / 66 held-out test | C0 (12 parameters), C1 (14); 4 tasks, 2 qualified and 2 unresolved |

One simulated RWGS **run** has one initial composition and four **species time
courses**. There are 1,188 non-initial observations in the 3-run design and
15,840 in the 40-run fitting design; validation has 3,960. The designs differ
in composition coverage as well as count. They are not a one-factor experiment.

- `data/`: actual observations, original times and initial states, source-record
  IDs, operating conditions, fixed splits, masks and input groups. RWGS original
  array dtypes are preserved. Ni records retain raw conversion values and their
  explicit interpretation. Methanol synthesis records retain printed tokens, raw fields,
  working interpretations and missing observations as `null` plus a mask.
- `configs/`: original seeds, budgets, initialization and selection rules,
  candidate definitions, units and thermodynamic conventions. Data generation
  parameters are evaluation references only, never optimizer inputs.
- `src/`: the existing scientific kernels, shared CPU solver, case adapters and
  existing SI decomposition. Only import roots, data paths and frozen-data
  loaders were simplified. The shared solver files match the archived RWGS
  and methanol synthesis source snapshots byte-for-byte.
- `reference/`: compact selected parameters, qualification/task records,
  predictions for comparison, conditional-refit anchors and Jacobians. Each
  Jacobian has parameter order, center, scale, bounds and residual convention.
  `provenance.json` links inputs to original record paths and SHA-256 hashes;
  these paths are provenance, not runtime dependencies. `package_hashes.json`
  checks the release's scientific inputs and code.

The manuscript case is named **Methanol synthesis over Cu/ZnO/Al₂O₃**;
`methanol` is its short software identifier. Original source paths and version
strings remain verbatim in provenance records only, for traceability.

### Units, objectives and sources

**RWGS:** ideal-gas activities `p_i / 1 bar`, gas-phase water, 700 K, constant
temperature and volume; rate and time retain inherited model units, not a
catalyst-calibrated physical time basis. The NIST WebBook Shomate coefficients
for CO2, H2, CO and H2O(g) are retained in `thermo_reduction.py`; they give
`K_eq(700 K) = 0.10589056838447206`. The generator uses
`k_f=0.0813`, forward powers `(1, 0.5, 0, 0)`, `K4=16.3`, and denominator
`(1 + K4*a_H2O/a_H2)^2`. Reverse parameters are derived, not separately fitted.
Noise is `exp(0.03*z - 0.03^2/2)` with exact initial states. The standardized
log residual is `(ln(observed)-ln(predicted)+sigma^2/2)/(sigma*sqrt(N))`;
`F=0.5*r.T*r`. The 3-run noise seed is 12001; the independent 40/10 design
uses master seed 314159 and the archived child streams. No new observations
are generated by reproduction. Sources:
[CO2](https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=1&Units=SI),
[H2](https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1&Units=SI),
[CO](https://webbook.nist.gov/cgi/cbook.cgi?ID=C630080&Mask=1&Units=SI),
[H2O](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=1&Units=SI).

**Ni/CeO2:** Onrubia-Calvo et al., *Kinetics, Model Discrimination, and
Parameters Estimation of CO2 Methanation on Highly Active Ni/CeO2 Catalyst*,
IECR 61 (2022), 10419-10435,
[DOI: 10.1021/acs.iecr.2c00164](https://doi.org/10.1021/acs.iecr.2c00164), SI S7.
The 124 source rows were transcribed and checked in the original project.
Printed `X=0.02` is interpreted as a conversion fraction (2%) despite a percent
header; no observation was rescaled or corrected to fit a model. Pressure is
bar, temperature K, contact coordinate g_cat h/mol_CO2, and rate mol/(g_cat h).
The empirical equilibrium relation retains its explicit kJ-to-J conversion.
The internal residual is `(X_pred-X_obs)/sqrt(N)` and
`RMSE_pp=100*sqrt(2F)`. Printing resolution is not a measured standard deviation.
The 5-percentage-point training utility gate is not a known-noise criterion.
Whole input/temperature groups define the split. All water cofeed points are
in fitting. Literature-reference initialization has prior exposure to the full
source dataset and is never pooled with ordinary initialization as independent
recovery evidence. Conditional-refit A anchors ordinary A/2101; B anchors
reference-initialized B/0. The opposite B offset was outside the original bound.

**Methanol synthesis over Cu/ZnO/Al₂O₃:** experimental SI Table S1 from
[Reaction Chemistry & Engineering, DOI: 10.1039/D1RE00040C](https://doi.org/10.1039/D1RE00040C),
with [Correction, DOI: 10.1039/D1RE90031E](https://doi.org/10.1039/D1RE90031E).
The 359 extracted rows were checked against the supplied bGRNN dataset
([J. Catal., DOI: 10.1016/j.jcat.2026.116958](https://doi.org/10.1016/j.jcat.2026.116958)).
Measured temperatures are **210-260 degC**, pressures 41/61 bar. The model's
thermodynamic validity interval is not the measured interval. The working
interpretation uses wet outlets, zero inlet methanol/water, preserved raw
printed nitrogen and balance-derived interpreted nitrogen. Pressure is Pa,
flow mol/s, catalyst kg, rate mol/(kg_cat s), standard pressure 100000 Pa,
and effective temperature slopes kJ/mol. The SI Eq.19 Gibbs law, its derived
cycle closure and Peng-Robinson gas-root fugacities are preserved; no conflicting
enthalpy row is substituted. Only CO, CO2 and methanol outlets are observed.
`F=0.5*mean((100*(predicted-observed))^2)` over observed entries only.
Input-only groups (not asserted experimental batch identities) preserve the
frozen 227/66/66 allocation. The test set is replayed only with the already
locked parameters, without tuning or reselection.

## Outputs and quick checks

Each case writes prediction/error source tables, physical parameter coordinates,
explicit expressions, archived candidate/task summaries and SVD tables.
`summary.json` records actual replay cost; `verification.json` contains every
comparison, tolerance and pass/fail result. Local singular vectors may reverse
overall sign. Jacobians contain data residuals only, not damping rows.
`J_scaled = J_physical * parameter_scale`; physical-coordinate and scaled
spectra are reported separately. Multiplication by `sqrt(N)` removes observation
count normalization, not observational weighting. No new derivative accuracy
or parameter-identifiability claim follows from these matrix operations.

Quick checks at the archived parameters:

| Quantity | Reference |
|---|---:|
| RWGS 3-run net numerator / reference, first limited-design initial composition | 12668.81559107171 |
| RWGS 3-run site availability / reference, same composition | 0.00008754262792704052 |
| Ni ordinary fixed-order A validation RMSE (percentage points) | 5.255000030337525 |
| Ni ordinary fixed-order B validation RMSE (percentage points) | 5.975076700375141 |
| Methanol synthesis: C1/C0 methanol-source ratio, common C1 outlet of record 99 | 1.8402910284840952 |
| Methanol synthesis: H2 site contribution, C0 at that state | +0.499554147473925 |
| Methanol synthesis: H2 site contribution, C1 at that state | -0.40923074295096507 |

The RWGS opposing factors are approximately four orders of magnitude, not
four orders of predictive error. Its common composition is the first archived
3-run initial state; all coordinates are exported. Methanol synthesis common states use full
precision archived mole fractions/PR activities at 503.15 K and 41 bar, not
rounded manuscript activities. Their parent outlet predictions are independently
replayed before the same-state decomposition.

Comparison tolerances are fixed in `scripts/reproduce.py`: maximum absolute
prediction differences 2e-8 (RWGS activities) or 2e-7 (Ni fractions/methanol mole
fractions); RMSE differences 2e-5 percentage points; RWGS standardized MSE
2e-7; same-state algebraic quantities 1e-10 absolute plus 1e-10 relative;
saved singular values 1e-12 absolute plus 1e-10 relative; sign-aligned weakest
vectors 1e-7 absolute. These are reproduction tolerances, **not modified fitting
or numerical-qualification thresholds**. Differences are reported, not hidden
by replacing a result or reranking candidates.

Release verification copied this repository to a fresh temporary directory,
unset `PYTHONPATH`, set `OPENBLAS_NUM_THREADS=1`, and ran the default command.
**239 comparisons passed, none failed**, in approximately 10 seconds on the
available CPU host. Compared saved predictions and reported RMSEs matched
exactly; the largest objective difference was `2.9501956433364285e-11`.
Imported scientific modules were checked to reside inside the relocated
repository. This is a fixed-parameter replay, not a new optimization,
qualification test, or identifiability result. Optional fitting commands were
syntax/import checked but not executed.

## Optional original estimation routes

These commands are opt-in, potentially expensive, and were **not executed**
during packaging. Use a new output directory. No new fitted point replaces a
paper result, and no validation/test response participates in these commands.

```bash
python scripts/fit.py --case rwgs --design 3 --candidate S_4 --output outputs/rwgs_refit3
python scripts/fit.py --case rwgs --design 40 --candidate S_4 --output outputs/rwgs_refit40
python scripts/fit.py --case ni --candidate A --stage thermal --output outputs/ni_A
python scripts/fit.py --case ni --refit PROFILE_A_-1 --output outputs/ni_conditional
python scripts/fit.py --case methanol --output outputs/methanol_refit
```

RWGS defaults to all five original starts, 2000 calls each including 64 reserved
for diagnostics; `--seed` accepts only original seeds. Each of the original
16 structures is available by its frozen ID. Ni stages use fresh data-scale
or explicit literature-reference initializations, never the preceding stage's
fitted parameters: 400/1000/1000 calls, each with 64 reserved; conditional
refits use 200 including 64 reserved and their declared archived anchors.
Ni outputs select representatives separately by initialization category.
Methanol synthesis uses both original seeds per candidate, 300 optimization calls or 1800 s,
then up to 64 checks at one optimizer-converged point per candidate, selected
by training objective. No qualification fallback is added. Physical bounds,
scales and numerical thresholds remain in the original model/solver code and
recorded configs. Full batch orchestration and old review infrastructure are
intentionally excluded; the default replay already covers the frozen batch.

## Provenance, limitations and release checks

No historical result or lock was edited. Original task hashes were checked
against the training locks during assembly. The following specific provenance
issue is retained rather than silently repaired: the October 28 local
compensation script calls RWGS seed **2101** the selected 3-run representative,
whereas the frozen training lock selects **2203**. This package uses 2203 for
representative predictions and includes 2101 as a separately labeled historical
sensitivity point; their arrays and source IDs are not mixed.

The final assembled manuscript and full updated SI document were not
identifiable in the workspace. The available Section 3 draft, revised RWGS
section, updated methanol synthesis numerical SI (October 9), Ni output revisions and frozen
records were checked. Thus, a full line-by-line audit of the final submission
cannot be claimed. No missing data were fabricated.

The original SBS workspace has no identified project-wide redistribution
license. The authors must approve one before public release; this repository
does not invent licensing rights for the source experimental tables.
`THIRD_PARTY_LICENSE.txt` retains the supplied VotsmeierGroup MIT notice for
the bGRNN-derived thermodynamics configuration. Original paper/SI PDFs,
manuscripts, environments, caches, GPU dependencies and optimization histories
are not distributed.
