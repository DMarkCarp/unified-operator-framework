## Simulation Specification

This section pins down the data-generating process, grids, estimators, and metrics. All four theoretical claims (exactness under discrete measure, convergence as grid densifies, equivalence of vectorized multivariate regression to the discrete-measure operator representation, and the role of conditioning/regularization) must be illustrated using the setup below. Do not substitute alternative DGPs or estimators without flagging the deviation.

Throughout, the simulation is framed as an operator-estimation problem. The target is the integral operator `T_beta : L^2([0,1]) -> L^2([0,1])` defined by `(T_beta X)(t) = integral beta(s,t) X(s) ds`. Discretization replaces the continuous Lebesgue measure with a discrete probability measure `mu` supported on the grid; the discrete operator `T_beta^mu` is the exact representation of `T_beta` *under that measure*. Vectorized multivariate regression estimates `T_beta^mu` directly, with no model change — only the underlying measure changes. All conclusions are to be interpreted in this language: differences in performance reflect properties of `mu` (grid density, weights), the design (conditioning, sample size), and the estimator (OLS vs ridge), **not** different regression models.

### 1. Setting

Function-on-function regression on the unit interval. Both predictor and response are observed on a common grid for simplicity; the framework extends to distinct grids but that is not the focus here.

- Domain: `s, t in [0, 1]`
- Predictor: `X(s)`, a random function on `[0,1]`
- Response: `Y(t) = integral_0^1 beta(s, t) X(s) ds + epsilon(t)`
- True coefficient surface:
  `beta(s, t) = sin(2 * pi * s) * cos(2 * pi * t) + s * t`
  This is smooth, non-separable, and has both oscillatory and polynomial structure — enough to make discretization error visible without being pathological.

### 2. Predictor process

`X(s)` is generated as a truncated Karhunen-Loeve expansion:

```
X(s) = sum_{k=1}^{K} xi_k * phi_k(s)
```

with:
- `K = 50` basis functions
- `phi_k(s) = sqrt(2) * sin((k - 0.5) * pi * s)` (Fourier sine basis)
- `xi_k ~ N(0, lambda_k)` independently, with `lambda_k = k^(-2)` (smooth-decay eigenvalues)

This gives smooth but non-trivial predictors and a controllable effective dimension.

### 3. Noise

Additive iid Gaussian noise on the response grid:
- `epsilon(t_j) ~ N(0, sigma^2)` independently across `j`
- Default `sigma` is calibrated to a target SNR of 5 (signal variance / noise variance), computed from a pilot run with `n = 1000`, `p = 200`. Report the chosen `sigma` in the summary.

### 4. Discretization as a discrete measure

The grid plus weights together specify a discrete measure `mu` on `[0,1]`; this measure is the central object, not a numerical approximation device. For grid size `p`:

- Atoms: `s_i = (i - 0.5) / p` for `i = 1, ..., p` (and analogously `t_j` on the response side)
- Weights: `w_i = 1/p`

Equivalently, `mu = (1/p) sum_i delta_{s_i}`, the uniform discrete probability measure on the midpoint grid. The integral operator under `mu` is

```
(T_beta^mu X)(t) = sum_i w_i * beta(s_i, t) * X(s_i)
```

and this is **exact** — there is no approximation when `X` and `beta` are evaluated under `mu`.

This simulation uses a uniform discrete probability measure `mu` for simplicity and clarity of exposition. The framework itself allows any discrete measure with arbitrary positive weights `w_i > 0` and atoms `s_i` placed anywhere in `[0,1]` — including trapezoidal weights, Gauss-Legendre nodes, observation-driven non-uniform sampling, or weights estimated from data. All theoretical claims and the implementation pattern (square-root-weight scaling, vectorized regression, kernel recovery) extend to this general setting; only the numerical values of the weights change. We commit to the uniform midpoint choice throughout this simulation for consistency; do not switch mid-experiment.

Claim (2) — convergence of the discrete operator to the continuous one — corresponds in this simulation to an *increasingly fine discretization of the domain* as `p` grows. We document this convergence empirically through the rate at which operator error decreases with `p`; we do not implement or claim a formal weak-convergence argument here.

### 5. Sweeps

Three nested sweeps drive the figures:

**Sweep A — Grid density (drives operator-error and convergence claims):**
- `p in {10, 20, 40, 80, 160, 320}`
- Sample size fixed at `n = 500`
- Monte Carlo replications: `R = 100` per grid size

**Sweep B — Sample size at fixed grid (drives prediction-error claim):**
- `n in {100, 250, 500, 1000, 2000}`
- Grid size fixed at `p = 80`
- Monte Carlo replications: `R = 100` per cell

**Sweep C — Conditioning (drives ridge-vs-OLS claim):**
- `p in {40, 80, 160, 320}`
- `n in {p/2, p, 2*p}` (so the design ranges from underdetermined to well-posed)
- Monte Carlo replications: `R = 50` per cell

A separate held-out test set of size `n_test = 500` is generated fresh each replication, on the same grid as training.

### 6. Estimators

Only two estimators are needed; the point is not to compare methods but to expose the operator-equivalence and conditioning behavior.

**OLS (vectorized multivariate regression):**

Construct the design matrix `X_mat` of shape `(n, p)` with entries
```
X_mat[k, i] = sqrt(w_i) * X_k(s_i) = (1/sqrt(p)) * X_k(s_i)
```
The square-root-of-weights scaling is essential and serves a specific purpose: it makes the empirical Gram matrix `X_mat^T X_mat / n` an unbiased estimator of the population Gram operator under the discrete L^2(mu) inner product,
```
<f, g>_{L^2(mu)} = sum_i w_i * f(s_i) * g(s_i)
```
Without this scaling, the Gram matrix would be on the wrong scale by a factor of `p`, and the resulting OLS estimate of `B` would not directly correspond to the discrete-measure operator `T_beta^mu`. With the scaling in place, vectorized OLS *is* — not merely approximates — the empirical estimator of `T_beta^mu`. This is the operational content of claim (3).

Similarly stack the responses into `Y_mat` of shape `(n, p)` with `Y_mat[k, j] = Y_k(t_j)` (no weight scaling on the response side, since the response inner product is handled at the metric stage).

Estimate
```
B_hat = (X_mat^T X_mat)^(-1) X_mat^T Y_mat
```
when well-posed; use `numpy.linalg.lstsq` with default `rcond` otherwise.

**Recovery relationship between `B_hat` and `beta_hat`.** Because the predictor matrix entries are scaled as `X_mat[k, i] = sqrt(w_i) * X_k(s_i)`, the matrix `B_hat` is the regression coefficient on *weight-scaled* predictors. The kernel of the discrete operator is recovered by inverting that scaling:
```
beta_hat(s_i, t_j) = (1 / sqrt(w_i)) * B_hat[i, j]
```
For the uniform measure used here, `w_i = 1/p`, so `1/sqrt(w_i) = sqrt(p)` and equivalently
```
beta_hat(s_i, t_j) = sqrt(p) * B_hat[i, j].
```
The general form `(1/sqrt(w_i)) * B_hat[i, j]` is what extends to non-uniform discrete measures; the `sqrt(p)` form is the specialization to the uniform case. This recovered kernel is what gets compared to the true `beta(s_i, t_j)` in the operator-error metric below.

**Ridge:**
- Same vectorized form: `B_hat_ridge = (X_mat^T X_mat + lambda * I)^(-1) X_mat^T Y_mat`
- `lambda` selected by 5-fold CV on the training set, searching over `lambda in 10^np.linspace(-6, 2, 25)`.

No FPCR, no neural methods, no basis projection. Raw discretization only — that is the entire point.

### 7. Metrics

All metrics computed per replication, then summarized as mean and Monte Carlo standard error across replications. Two distinct error notions are tracked, both grounded in the operator framework: error in the *kernel* (coefficient surface) and error in the *operator action* on test inputs.

**Coefficient ISE (kernel-level error, computed under the discrete `L^2(mu x mu)` norm):**
```
ISE = sum_{i,j} w_i * w_j * (beta_hat(s_i, t_j) - beta(s_i, t_j))^2
```
For the uniform measure used here, `w_i = w_j = 1/p`, so this reduces to `(1/p^2) * sum_{i,j} (beta_hat(s_i, t_j) - beta(s_i, t_j))^2`. This is the squared `L^2(mu x mu)` distance between estimated and true kernels. It measures how well the estimator recovers the kernel of `T_beta^mu` itself, ignoring how that kernel acts on inputs.

**Operator error (action-level estimation error on test data, computed under the discrete `L^2(mu)` norm):**
```
OpErr^2 = (1/n_test) * sum_k || (T_{beta_hat}^mu X_k^test) - (T_beta^mu X_k^test) ||^2_{L^2(mu)}
        = (1/n_test) * sum_k * sum_j w_j * ( Y_signal_hat_k(t_j) - Y_signal_k(t_j) )^2
```
where:
- `Y_signal_k(t_j) = sum_i w_i * beta(s_i, t_j) * X_k^test(s_i)` is the **noiseless** true operator output under `mu` (no `epsilon` added),
- `Y_signal_hat_k(t_j) = sum_i w_i * beta_hat(s_i, t_j) * X_k^test(s_i)` is the **noiseless** estimated operator output under `mu`,
- the outer norm is the discrete `L^2(mu)` norm on the response side, with weights `w_j` matching the response-side discrete measure (uniform `1/p` here).

`OpErr` measures *estimation error*: it compares two estimators of the same discrete operator at the same `mu`. It does **not** measure the discretization gap between `T_beta^mu` and the continuous `T_beta` — both `T_beta_hat^mu` and `T_beta^mu` live at the same grid `mu`. The discretization gap is the separate `DiscErr` metric below. Report `OpErr` alongside `ISE` for every replication; the two are complementary kernel-vs-action diagnostics of the same fitted operator.

**Discretization error (operator-vs-continuous gap, computed under the discrete `L^2(mu_ref)` norm using a fine reference grid `mu_ref`):**
```
DiscErr^2 = (1/n_test) * sum_k || (T_beta^{mu_ref} X_k^test) - (T_beta^mu X_k^test) ||^2_{L^2(mu_ref)}
          = (1/n_test) * (1/p_ref) * sum_k * sum_{j_ref} ( Y_ref_k(t_{j_ref}) - Y_coarse_k(t_{j_ref}) )^2
```
where:
- `p_ref = 2000` is the reference grid size; midpoint discretization at `p_ref` approximates the continuous operator `T_beta` to leading order with error `O(p_ref^(-2))` ≪ the gap at any `p` in the sweeps,
- `Y_ref_k(t_{j_ref}) = sum_{i_ref} (1/p_ref) * beta(s_{i_ref}, t_{j_ref}) * X_k^test(s_{i_ref})` is the reference-grid operator output at the test input,
- `Y_coarse_k(t_{j_ref}) = sum_i (1/p) * beta(s_i, t_{j_ref}) * X_k^test(s_i)` is the coarse-grid operator output evaluated on the **same** (reference) `t`-axis,
- `X_k^test` is evaluated at both the coarse and reference grids from the same KL coefficients, so the only source of difference between `Y_ref` and `Y_coarse` is the choice of measure on the `s`-axis.

`DiscErr` measures the discretization gap `|| T_beta^mu - T_beta ||` (with `T_beta` approximated by `T_beta^{mu_ref}`). It does **not** depend on `beta_hat`, on training data, or on `n` — it is a property of `(beta, mu, mu_ref)` alone. For smooth `beta` under uniform midpoint discretization the rate is `O(p^(-2))` (squared Riemann-rule error). This is the metric that directly captures claim (2). It is the *primary* diagnostic for Sweep A; `OpErr` and `ISE` are reported alongside as estimation-error context.

**Prediction error (test RMSE on noisy responses):**
```
RMSE = sqrt( (1/(n_test * p)) * sum_{k, j} (Y_test_k(t_j) - Y_hat_k(t_j))^2 )
```
where `Y_test_k` includes the test-side noise. This is what a practitioner would measure; it equals `OpErr^2 + sigma^2` in expectation, providing an internal consistency check.

**Conditioning:**
- `kappa = numpy.linalg.cond(X_mat^T X_mat)` reported per replication.
- For Sweep C, report condition number alongside both OLS and ridge prediction RMSE *and* operator error in the same table so the reader can see both metrics blow up with `kappa` for OLS while ridge stays controlled.

### 8. Reproducibility

- Top-level seed: `RNG_SEED = 20260502` (today's date). Every replication derives its seed deterministically as `RNG_SEED + 1000 * sweep_id + replication_index`.
- Python: 3.11 or newer.
- Pin versions in a `requirements.txt`: `numpy>=1.26`, `scipy>=1.11`, `pandas>=2.1`, `scikit-learn>=1.4`, `matplotlib>=3.8`.
- All randomness routed through `numpy.random.default_rng(seed)` — no legacy `numpy.random.seed`.

### 9. Figure conventions

- Style: `matplotlib` defaults with the Auburn palette — navy `#03244D` for primary lines, burnt orange `#DD550C` for secondary/contrast, neutral grey `#5F6A6A` for reference lines and OLS-when-ridge-is-primary.
- All sweep figures use log-log axes where the predicted scaling is a power law (operator error vs `p`, RMSE vs `n`).
- Error bars or shaded bands show Monte Carlo standard error (not standard deviation).
- Each figure saved at 300 dpi as both PNG and PDF, with `bbox_inches='tight'`.
- `true_beta_surface.png` is a filled contour plot of `beta(s,t)`; `estimated_B_heatmap.png` shows the ridge estimate `B_hat_ridge` (with CV-selected `lambda`) from one representative replication at `p = 80, n = 500` for visual comparison. Ridge — not OLS — is used here because at `p > K` (the KL truncation of the predictor process), the OLS minimum-norm pseudoinverse is dominated by noise across the unidentifiable directions and is not visually representative of the operator the framework recovers; the ridge estimate is. The heatmap is rescaled by `sqrt(p)` per the recovery formula in section 6 so that it is on the same scale as `beta`.

### 10. What the figures must show (acceptance criteria)

All criteria are stated in operator-framework language. "Discretization error" refers to `DiscErr` from section 7 (the `T_beta^mu` vs `T_beta` gap), and "operator error" to `OpErr` (the `T_beta_hat^mu` vs `T_beta^mu` estimation error at fixed `mu`). The two are distinct quantities that the figures must keep distinct.

- `operator_error_vs_grid_size`: `DiscErr` is the primary curve and decreases with `p` at the rate predicted for the discrete-to-continuous operator gap on smooth kernels — roughly `p^(-2)` under uniform midpoint discretization. Log-log slope of `DiscErr` reported in the summary. `OpErr` and `sqrt(ISE)` are plotted on the same axes as secondary curves; they are *estimation*-error metrics at fixed `mu` and need not track the discretization rate, particularly when finite-sample variance dominates bias (which happens here for the chosen `n`, `sigma`, and `K`). The figure makes the kernel-vs-discretization distinction visible: `DiscErr` ↓ with `p` even as `OpErr` is dominated by variance.
- `prediction_error_by_grid_size` (really by sample size at fixed `p`): RMSE and `OpErr` both decrease as `n^(-1/2)` until floored by the irreducible noise variance (RMSE → `sigma`) and by the residual discretization gap (`OpErr` → 0 as `n` → ∞ at fixed `p`, while `DiscErr` is the `n`-independent `T_beta^mu` vs `T_beta` floor).
- `ridge_vs_ols_conditioning`: OLS `OpErr` and RMSE track `kappa` upward as `n/p` shrinks; ridge stays flat. Plot the operator-action metric on the y-axis, `kappa` on a secondary x-axis or as a colorbar. The contrast is not between two regression *models* — both are estimates of the same operator `T_beta^mu` — but between an unregularized and a regularized estimator of that one operator.
- `example_true_vs_predicted_curves`: 3-5 randomly chosen test response curves overlaid with their predictions at `p = 80, n = 500`. These are realizations of `T_{beta_hat}^mu X^test` versus `T_beta^mu X^test + epsilon`.
- `true_beta_surface` and `estimated_B_heatmap`: visually similar at `p = 80, n = 500` when the heatmap shows the ridge estimate (with CV-selected `lambda`) rescaled by `sqrt(p)`. With OLS at this `(n, p, K)` the heatmap is dominated by noise across the unidentifiable directions of the rank-`K` design and is *not* expected to be visually similar — that is a property of the unregularized estimator under the rank constraint, not a failure of the recovery formula. The recovery formula `(1/sqrt(w_i)) * B_hat[i, j]` itself is what makes the ridge heatmap line up on the same scale as `beta`.

### Internal validation rule (not for the summary)

The acceptance criteria above are diagnostics for the implementer, not findings for the summary. Specifically: if any of the qualitative patterns listed in section 10 fail to appear (`DiscErr` not decreasing with `p` at roughly `p^(-2)`, RMSE not following `n^(-1/2)`, ridge not stabilizing OLS in Sweep C, ridge heatmap not visually matching the true surface), treat this as evidence of an implementation bug — not as a substantive finding. Investigate, fix, and re-run before producing any output. Do **not** include statements like "the expected pattern did not appear" in the summary; the summary should report only the patterns observed in a correctly executed simulation. Note: `OpErr` not decreasing with `p` in Sweep A is **not** a bug and does **not** trigger this rule — `OpErr` measures estimation error at fixed `mu`, which need not track the `mu` → continuous discretization rate; `DiscErr` is the metric that does.

### 11. Interpretive framing for the summary

The Markdown summary in `results/simulation_summary.md` must explicitly frame all conclusions in operator-theoretic terms. Specifically:

- State that OLS and ridge are two estimators of the *same* operator `T_beta^mu`, not two competing regression models.
- Attribute the grid-size convergence (claim 2) to *increasingly fine discretization of the domain* as `p` grows. Do not invoke a formal weak-convergence argument; we have not implemented one. Frame it as: as `mu` becomes a finer discretization, `T_beta^mu` more closely represents the action of `T_beta` on the relevant function class.
- Attribute the `n^(-1/2)` regime to standard estimation-error scaling, and the floor to the irreducible gap between `T_beta^mu` and `T_beta` at finite `p` (i.e., to the choice and resolution of the discrete measure).
- Attribute the OLS-vs-ridge gap in Sweep C to conditioning of the empirical Gram operator under `mu`, controlled by ridge's spectral regularization — again, *one* operator, *two* estimators.
- Avoid framing any result as a model-comparison finding. The whole point of the paper is that the apparent diversity of "functional regression methods" collapses into one operator-estimation problem under different measures, designs, and regularizers.
