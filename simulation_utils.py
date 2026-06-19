"""Reusable utilities for the unified-operator-framework simulation.

Implements the data-generating process, OLS / ridge estimators, and metrics
exactly as specified in `simulation_specification.md`. The discrete-measure
framing is preserved throughout: predictors are scaled by `sqrt(w_i)` in the
design matrix, and the kernel is recovered via `(1/sqrt(w_i)) * B_hat[i, j]`.

All randomness goes through `numpy.random.default_rng(seed)`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


RNG_SEED = 20260502  # Top-level seed (today's date per spec).
K_DEFAULT = 50  # Number of KL basis functions for X(s).
P_REF = 2000  # Reference grid size used to approximate the continuous operator
              # in the discretization-error metric (DiscErr). p_ref >> any p in
              # the sweeps so T_beta^{mu_ref} ~ T_beta to leading order.

# Sweep ID assignments used in per-replication seed derivation:
#   seed = RNG_SEED + 1000 * sweep_id + replication_index
SWEEP_IDS = {"calibration": 0, "A": 1, "B": 2, "C": 3, "demo": 4}

# Auburn palette for figures.
AUBURN_NAVY = "#03244D"
AUBURN_ORANGE = "#DD550C"
NEUTRAL_GREY = "#5F6A6A"

# Ridge lambda search grid (log-spaced, 25 values from 1e-6 to 1e2).
RIDGE_LAMBDAS = 10.0 ** np.linspace(-6, 2, 25)


# ---------------------------------------------------------------------------
# Grid + true kernel
# ---------------------------------------------------------------------------

def midpoint_grid(p: int) -> np.ndarray:
    """Uniform midpoint grid s_i = (i - 0.5)/p, i = 1..p."""
    return (np.arange(1, p + 1) - 0.5) / p


def true_beta_surface(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """beta(s, t) = sin(2 pi s) cos(2 pi t) + s t evaluated on a grid.

    Returns a matrix of shape (len(s), len(t)) with rows indexing s and columns
    indexing t.
    """
    S, T = np.meshgrid(s, t, indexing="ij")
    return np.sin(2 * np.pi * S) * np.cos(2 * np.pi * T) + S * T


# ---------------------------------------------------------------------------
# Predictor process: truncated Karhunen-Loeve expansion
# ---------------------------------------------------------------------------

def kl_basis(s: np.ndarray, K: int = K_DEFAULT) -> np.ndarray:
    """Fourier-sine basis phi_k(s) = sqrt(2) sin((k - 0.5) pi s), k = 1..K.

    Returns a matrix of shape (len(s), K).
    """
    k = np.arange(1, K + 1)
    return np.sqrt(2.0) * np.sin(np.outer(s, (k - 0.5) * np.pi))


def kl_eigenvalues(K: int = K_DEFAULT) -> np.ndarray:
    """Eigenvalues lambda_k = k^(-2) for k = 1..K."""
    return np.arange(1, K + 1, dtype=float) ** (-2.0)


def sample_xi(rng: np.random.Generator, n: int,
              K: int = K_DEFAULT) -> np.ndarray:
    """Draw n KL coefficient vectors xi ~ N(0, diag(lambda_k)). Shape (n, K).

    Decoupling xi from the grid lets the same predictor sample be evaluated on
    multiple grids — required by the discretization-error metric, which needs
    X^test on both the coarse grid (p) and a fine reference grid (p_ref).
    """
    eigvals = kl_eigenvalues(K)
    return rng.standard_normal((n, K)) * np.sqrt(eigvals)


def evaluate_X(xi: np.ndarray, s: np.ndarray,
               K: int | None = None) -> np.ndarray:
    """Evaluate X_k(s_i) = sum_m xi[k, m] phi_m(s_i) on grid s. Shape (n, p)."""
    if K is None:
        K = xi.shape[1]
    Phi = kl_basis(s, K)  # (p, K)
    return xi @ Phi.T  # (n, p)


def sample_X_curves(rng: np.random.Generator, n: int, s: np.ndarray,
                    K: int = K_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Draw n predictor curves on grid s as a truncated KL expansion.

    Returns (X_curves, xi) where X_curves has shape (n, p) and xi has shape
    (n, K). Exposing xi enables re-evaluation on a different grid (used by
    the discretization-error metric).
    """
    xi = sample_xi(rng, n, K)
    X_curves = evaluate_X(xi, s, K)
    return X_curves, xi


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    X_curves: np.ndarray  # (n, p) raw predictor evaluations X_k(s_i)
    Y_signal: np.ndarray  # (n, p) noiseless responses (T_beta X_k)(t_j)
    Y_noisy: np.ndarray   # (n, p) Y_signal + epsilon
    s_grid: np.ndarray    # (p,)
    t_grid: np.ndarray    # (p,)
    beta_grid: np.ndarray  # (p, p), rows = s, cols = t
    xi: np.ndarray        # (n, K) KL coefficients used to draw X_curves


def generate_dataset(rng: np.random.Generator, n: int, p: int, sigma: float,
                     K: int = K_DEFAULT) -> Dataset:
    """Generate a fresh function-on-function dataset on the uniform midpoint grid.

    The discrete operator action is computed as
        Y_signal[k, j] = sum_i w_i * beta(s_i, t_j) * X_k(s_i)
    with w_i = 1/p; this is exact under the discrete measure mu.
    """
    s_grid = midpoint_grid(p)
    t_grid = s_grid  # same grid on both sides (spec section 1)
    X_curves, xi = sample_X_curves(rng, n, s_grid, K)  # (n, p), (n, K)
    beta_grid = true_beta_surface(s_grid, t_grid)  # (p, p)
    # Y_signal[k, j] = (1/p) * sum_i beta[i, j] * X_curves[k, i]
    Y_signal = (X_curves @ beta_grid) / p
    eps = rng.standard_normal((n, p)) * sigma
    Y_noisy = Y_signal + eps
    return Dataset(X_curves=X_curves, Y_signal=Y_signal, Y_noisy=Y_noisy,
                   s_grid=s_grid, t_grid=t_grid, beta_grid=beta_grid, xi=xi)


def build_design(X_curves: np.ndarray, p: int) -> np.ndarray:
    """X_mat[k, i] = sqrt(w_i) * X_k(s_i) = (1/sqrt(p)) * X_k(s_i).

    The sqrt-weight scaling makes X_mat^T X_mat / n an unbiased estimator of
    the population Gram operator under the discrete L^2(mu) inner product.
    """
    return X_curves / np.sqrt(p)


# ---------------------------------------------------------------------------
# Sigma calibration (target SNR = 5 from a pilot run)
# ---------------------------------------------------------------------------

def calibrate_sigma(target_snr: float = 5.0, n_pilot: int = 1000,
                    p_pilot: int = 200, K: int = K_DEFAULT,
                    seed: int | None = None) -> dict:
    """Pick sigma so that signal variance / noise variance = target_snr.

    Computes Var(Y_signal) over all (k, j) entries from a single pilot draw
    of size (n_pilot, p_pilot) and returns the calibrated sigma along with
    diagnostic quantities.
    """
    if seed is None:
        seed = RNG_SEED + 1000 * SWEEP_IDS["calibration"]
    rng = np.random.default_rng(seed)
    s = midpoint_grid(p_pilot)
    X_curves, _ = sample_X_curves(rng, n_pilot, s, K)
    beta = true_beta_surface(s, s)
    Y_signal = (X_curves @ beta) / p_pilot
    signal_var = float(Y_signal.var())
    sigma = float(np.sqrt(signal_var / target_snr))
    return {
        "sigma": sigma,
        "signal_var": signal_var,
        "target_snr": target_snr,
        "n_pilot": n_pilot,
        "p_pilot": p_pilot,
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def fit_ols(X_mat: np.ndarray, Y_mat: np.ndarray) -> np.ndarray:
    """OLS on the weight-scaled design via `np.linalg.lstsq` with default rcond.

    Spec section 6 prescribes the explicit normal equations when well-posed
    and `lstsq(rcond=default)` otherwise. Because the predictor is a K=50 KL
    truncation, X_mat has rank min(n, p, K) = K for any p > K, so the Gram
    is rank-deficient at p > 50 and `np.linalg.solve` silently returns
    garbage rather than raising. lstsq via SVD with the default cutoff
    handles both regimes uniformly: it equals the direct solve to machine
    precision when full-rank, and gives the minimum-norm pseudoinverse
    solution when rank-deficient.
    """
    return np.linalg.lstsq(X_mat, Y_mat, rcond=None)[0]


def ridge_path_from_svd(U: np.ndarray, s_vals: np.ndarray, Vt: np.ndarray,
                        Y_mat: np.ndarray,
                        lambdas: Sequence[float]) -> dict[float, np.ndarray]:
    """Compute B_hat for each lambda from a precomputed thin SVD of X_mat.

    Identity used: with X = U S V^T (thin), B_hat(lambda) = V * diag(s_k /
    (s_k^2 + lambda)) * U^T Y. This handles both n >= p and n < p (returning
    the minimum-norm solution as lambda -> 0).
    """
    UtY = U.T @ Y_mat  # (r, q)
    out: dict[float, np.ndarray] = {}
    for lam in lambdas:
        d = s_vals / (s_vals ** 2 + lam)  # (r,)
        out[float(lam)] = Vt.T @ (d[:, None] * UtY)
    return out


def ridge_cv_select(X_mat: np.ndarray, Y_mat: np.ndarray,
                    lambdas: Sequence[float] = RIDGE_LAMBDAS,
                    n_folds: int = 5,
                    rng: np.random.Generator | None = None
                    ) -> tuple[float, np.ndarray, np.ndarray]:
    """5-fold CV over `lambdas`, refit on full training data via single SVD.

    Per fold: one SVD on the fold's training subset, then all 25 lambda
    solutions computed analytically from that SVD. Final fit reuses one more
    SVD on the full training data. Returns (best_lambda, B_hat, cv_errs).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = X_mat.shape[0]
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    lambdas = np.asarray(lambdas, dtype=float)
    cv_errs = np.zeros(len(lambdas))

    for f in range(n_folds):
        val_idx = folds[f]
        train_idx = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        X_tr, Y_tr = X_mat[train_idx], Y_mat[train_idx]
        X_va, Y_va = X_mat[val_idx], Y_mat[val_idx]
        U, s_vals, Vt = np.linalg.svd(X_tr, full_matrices=False)
        Bs = ridge_path_from_svd(U, s_vals, Vt, Y_tr, lambdas)
        for li, lam in enumerate(lambdas):
            Y_pred = X_va @ Bs[float(lam)]
            cv_errs[li] += float(((Y_va - Y_pred) ** 2).mean())
    cv_errs /= n_folds

    best_idx = int(np.argmin(cv_errs))
    best_lam = float(lambdas[best_idx])

    U, s_vals, Vt = np.linalg.svd(X_mat, full_matrices=False)
    B_best = ridge_path_from_svd(U, s_vals, Vt, Y_mat, [best_lam])[best_lam]
    return best_lam, B_best, cv_errs


# ---------------------------------------------------------------------------
# Recovery + metrics
# ---------------------------------------------------------------------------

def recover_beta_hat(B_hat: np.ndarray, p: int) -> np.ndarray:
    """beta_hat(s_i, t_j) = (1/sqrt(w_i)) * B_hat[i, j] = sqrt(p) * B_hat[i, j]
    for the uniform measure used here.
    """
    return np.sqrt(p) * B_hat


# ---------------------------------------------------------------------------
# Discretization error (operator-vs-continuous gap, spec section 7)
# ---------------------------------------------------------------------------
#
# DiscErr^2 = (1/n_test) sum_k || (T_beta^{mu_ref} X_k) - (T_beta^mu X_k) ||^2_{L^2(mu_ref)}
#           = (1/(n_test * p_ref)) sum_{k, j_ref} (Y_ref - Y_coarse)^2
#
# where Y_ref is computed on the fine reference grid (p_ref) and Y_coarse is
# the coarse-grid operator output evaluated at the *same* (reference) t-points.
# Both share the test inputs X_k, expressed via the KL coefficients xi.
#
# Trick: substituting X = xi @ Phi.T into both reduces the difference to
#     Y_ref - Y_coarse = xi @ (M_ref - M_p)
# with
#     M_ref = Phi_ref.T @ beta_ref / p_ref       shape (K, p_ref)
#     M_p   = Phi_p.T @ beta(s_p, t_ref) / p     shape (K, p_ref)
# Both are deterministic in (p, p_ref, K) and can be cached. Per-replication
# cost reduces to one (n_test, K) @ (K, p_ref) matmul (~ms).

_M_REF_CACHE: dict[tuple[int, int], np.ndarray] = {}
_M_P_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _get_M_ref(p_ref: int, K: int) -> np.ndarray:
    key = (p_ref, K)
    if key not in _M_REF_CACHE:
        s_ref = midpoint_grid(p_ref)
        Phi_ref = kl_basis(s_ref, K)                # (p_ref, K)
        beta_ref = true_beta_surface(s_ref, s_ref)  # (p_ref, p_ref)
        _M_REF_CACHE[key] = (Phi_ref.T @ beta_ref) / p_ref  # (K, p_ref)
    return _M_REF_CACHE[key]


def _get_M_p(p: int, p_ref: int, K: int) -> np.ndarray:
    key = (p, p_ref, K)
    if key not in _M_P_CACHE:
        s_p = midpoint_grid(p)
        s_ref = midpoint_grid(p_ref)
        Phi_p = kl_basis(s_p, K)                       # (p, K)
        beta_pt = true_beta_surface(s_p, s_ref)        # (p, p_ref)
        _M_P_CACHE[key] = (Phi_p.T @ beta_pt) / p      # (K, p_ref)
    return _M_P_CACHE[key]


def discretization_error(xi_test: np.ndarray, p: int,
                         p_ref: int = P_REF,
                         K: int = K_DEFAULT) -> float:
    """Discretization gap ||T_beta^mu - T_beta^{mu_ref}|| at the test inputs.

    Approximates the continuous operator T_beta by its discrete representation
    T_beta^{mu_ref} on a fine reference grid (p_ref). Returns the L^2(mu_ref)
    norm of the per-test difference between the coarse-grid and reference-grid
    operator outputs, averaged over the test draws. Independent of beta_hat.
    """
    M_ref = _get_M_ref(p_ref, K)
    M_p = _get_M_p(p, p_ref, K)
    Y_diff = xi_test @ (M_p - M_ref)  # (n_test, p_ref)
    return float(np.sqrt((Y_diff ** 2).mean()))


def compute_metrics(B_hat: np.ndarray, beta_grid: np.ndarray,
                    test: Dataset, p: int) -> dict[str, float]:
    """Compute ISE, OpErr, RMSE for one fitted B_hat against a test dataset.

    All under the discrete L^2(mu) / L^2(mu x mu) norms with w_i = 1/p.
    """
    beta_hat = recover_beta_hat(B_hat, p)  # (p, p)
    # Coefficient ISE under L^2(mu x mu): w_i * w_j = 1/p^2.
    ISE = float(((beta_hat - beta_grid) ** 2).sum() / (p * p))

    # Operator action on test inputs (noiseless): Y_signal_hat = X_test @ beta_hat / p
    Y_signal_hat = (test.X_curves @ beta_hat) / p  # (n_test, p)
    diff_signal = Y_signal_hat - test.Y_signal
    # OpErr^2 = (1/(n_test * p)) * sum_{k, j} diff^2  (uniform w_j = 1/p, mean over k)
    OpErr2 = float((diff_signal ** 2).mean())
    OpErr = float(np.sqrt(OpErr2))

    # Prediction RMSE on noisy test responses; Y_hat is noiseless predicted output.
    diff_noisy = Y_signal_hat - test.Y_noisy
    RMSE = float(np.sqrt((diff_noisy ** 2).mean()))

    return {"ISE": ISE, "OpErr": OpErr, "OpErr2": OpErr2, "RMSE": RMSE}


# ---------------------------------------------------------------------------
# Single-replication runner
# ---------------------------------------------------------------------------

def replication_seed(sweep_id: int, rep_idx: int) -> int:
    """Deterministic per-replication seed: RNG_SEED + 1000*sweep_id + rep_idx."""
    return RNG_SEED + 1000 * sweep_id + rep_idx


def run_replication(sweep_id: int, rep_idx: int, n: int, p: int, sigma: float,
                    estimator: str = "ols",
                    lambdas: Sequence[float] = RIDGE_LAMBDAS,
                    n_test: int = 500,
                    K: int = K_DEFAULT) -> dict:
    """Run one replication: generate train+test, fit estimator, compute metrics.

    `estimator` is "ols" or "ridge". For ridge, 5-fold CV is performed using
    a separate RNG stream (offset) so that fold assignments are deterministic
    from the same seed but independent of data sampling.
    """
    seed = replication_seed(sweep_id, rep_idx)
    rng = np.random.default_rng(seed)

    train = generate_dataset(rng, n, p, sigma, K=K)
    test = generate_dataset(rng, n_test, p, sigma, K=K)

    X_mat = build_design(train.X_curves, p)

    if estimator == "ols":
        B_hat = fit_ols(X_mat, train.Y_noisy)
        lam_used = 0.0
    elif estimator == "ridge":
        # Independent RNG stream for CV fold assignment so that data + fold
        # randomness do not collide on the same seed.
        rng_cv = np.random.default_rng(seed + 500_000)
        lam_used, B_hat, _ = ridge_cv_select(X_mat, train.Y_noisy, lambdas,
                                             n_folds=5, rng=rng_cv)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    kappa = float(np.linalg.cond(X_mat.T @ X_mat))
    metrics = compute_metrics(B_hat, train.beta_grid, test, p)
    metrics["DiscErr"] = discretization_error(test.xi, p)

    return {
        "sweep_id": sweep_id,
        "rep": rep_idx,
        "n": n,
        "p": p,
        "sigma": sigma,
        "estimator": estimator,
        "lambda": lam_used,
        "kappa": kappa,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Sweep runners
# ---------------------------------------------------------------------------

def _run_cells(cells: list[dict], R: int, sweep_id: int, sigma: float,
               n_jobs: int, K: int = K_DEFAULT) -> pd.DataFrame:
    """Generic sweep runner: iterates over a list of cell dicts and R reps each.

    Each cell dict must contain at least: n, p, estimator. May also contain
    a `cell_label` string for downstream grouping.
    """
    jobs = []
    for cell in cells:
        for rep in range(R):
            jobs.append((cell, rep))

    def _one(cell, rep):
        result = run_replication(
            sweep_id=sweep_id,
            rep_idx=rep,
            n=cell["n"],
            p=cell["p"],
            sigma=sigma,
            estimator=cell["estimator"],
            n_test=cell.get("n_test", 500),
            K=K,
        )
        result["cell_label"] = cell.get("cell_label", "")
        return result

    rows = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_one)(cell, rep) for cell, rep in jobs
    )
    return pd.DataFrame(rows)


def sweep_A_cells() -> list[dict]:
    """Sweep A: grid density, OLS, n=500, p in {10, 20, 40, 80, 160, 320}."""
    return [
        {"n": 500, "p": p, "estimator": "ols", "cell_label": f"p={p}"}
        for p in (10, 20, 40, 80, 160, 320)
    ]


def sweep_B_cells() -> list[dict]:
    """Sweep B: sample size, OLS, p=80, n in {100, 250, 500, 1000, 2000}."""
    return [
        {"n": n, "p": 80, "estimator": "ols", "cell_label": f"n={n}"}
        for n in (100, 250, 500, 1000, 2000)
    ]


def sweep_C_cells() -> list[dict]:
    """Sweep C: conditioning, OLS + ridge, p in {40,80,160,320}, n in {p/2,p,2p}."""
    cells = []
    for p in (40, 80, 160, 320):
        for ratio_label, n in (("n=p/2", p // 2), ("n=p", p), ("n=2p", 2 * p)):
            for est in ("ols", "ridge"):
                cells.append({
                    "n": n, "p": p, "estimator": est,
                    "cell_label": f"p={p},{ratio_label}",
                })
    return cells


def run_sweep_A(R: int, sigma: float, n_jobs: int = -1) -> pd.DataFrame:
    return _run_cells(sweep_A_cells(), R=R, sweep_id=SWEEP_IDS["A"],
                      sigma=sigma, n_jobs=n_jobs)


def run_sweep_B(R: int, sigma: float, n_jobs: int = -1) -> pd.DataFrame:
    return _run_cells(sweep_B_cells(), R=R, sweep_id=SWEEP_IDS["B"],
                      sigma=sigma, n_jobs=n_jobs)


def run_sweep_C(R: int, sigma: float, n_jobs: int = -1) -> pd.DataFrame:
    return _run_cells(sweep_C_cells(), R=R, sweep_id=SWEEP_IDS["C"],
                      sigma=sigma, n_jobs=n_jobs)


# ---------------------------------------------------------------------------
# Aggregation helper (mean + Monte Carlo SE per cell)
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame, group_keys: list[str],
              metric_cols: Iterable[str] = ("ISE", "OpErr", "OpErr2",
                                            "DiscErr", "RMSE", "kappa")) -> pd.DataFrame:
    """Compute mean and Monte Carlo SE (= std / sqrt(R)) per group."""
    agg = {}
    for col in metric_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_se"] = (col, lambda x: x.std(ddof=1) / np.sqrt(len(x)))
    out = df.groupby(group_keys, as_index=False).agg(**agg)
    out["R"] = df.groupby(group_keys).size().values
    return out


# ---------------------------------------------------------------------------
# Demo replication for figures (heatmap + example curves)
# ---------------------------------------------------------------------------

def demo_replication(sigma: float, n: int = 500, p: int = 80,
                     n_test: int = 500, K: int = K_DEFAULT,
                     estimator: str = "ridge",
                     lambdas: Sequence[float] = RIDGE_LAMBDAS) -> dict:
    """Run one replication at (n, p) for the heatmap and example-curves figures.

    Default estimator is ridge with CV-selected lambda. Per spec section 9
    (revised): at p > K (the predictor's KL truncation), OLS produces a
    minimum-norm pseudoinverse solution that is dominated by noise across the
    unidentifiable directions, so the regularized estimate is what is shown
    in the visual heatmap. Pass `estimator="ols"` to recover the unregularized
    representative if needed for diagnostic comparison.
    """
    seed = replication_seed(SWEEP_IDS["demo"], 0)
    rng = np.random.default_rng(seed)
    train = generate_dataset(rng, n, p, sigma, K=K)
    test = generate_dataset(rng, n_test, p, sigma, K=K)
    X_mat = build_design(train.X_curves, p)

    if estimator == "ols":
        B_hat = fit_ols(X_mat, train.Y_noisy)
        lam_used = 0.0
    elif estimator == "ridge":
        rng_cv = np.random.default_rng(seed + 500_000)
        lam_used, B_hat, _ = ridge_cv_select(X_mat, train.Y_noisy, lambdas,
                                             n_folds=5, rng=rng_cv)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    beta_hat = recover_beta_hat(B_hat, p)
    Y_signal_hat = (test.X_curves @ beta_hat) / p
    metrics = compute_metrics(B_hat, train.beta_grid, test, p)
    metrics["DiscErr"] = discretization_error(test.xi, p)
    return {
        "train": train, "test": test, "B_hat": B_hat, "beta_hat": beta_hat,
        "Y_signal_hat": Y_signal_hat, "metrics": metrics, "n": n, "p": p,
        "seed": seed, "estimator": estimator, "lambda": lam_used,
    }


# ---------------------------------------------------------------------------
# Plotting helpers (kept here so the notebook stays slim)
# ---------------------------------------------------------------------------

def _save_fig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")


def plot_true_beta_surface(out_dir: Path, p: int = 200) -> None:
    import matplotlib.pyplot as plt
    s = midpoint_grid(p)
    beta = true_beta_surface(s, s)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cf = ax.contourf(s, s, beta.T, levels=20, cmap="RdBu_r")
    ax.set_xlabel("s")
    ax.set_ylabel("t")
    ax.set_title(r"True $\beta(s, t) = \sin(2\pi s)\cos(2\pi t) + s t$")
    fig.colorbar(cf, ax=ax)
    _save_fig(fig, out_dir, "true_beta_surface")
    plt.close(fig)


def plot_estimated_B_heatmap(demo: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    p = demo["p"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    # Independent colorbars: the OLS pseudoinverse at p > K can have values
    # much larger than the truth, so a shared scale washes out beta.
    v_true = float(np.abs(demo["train"].beta_grid).max())
    v_hat = float(np.abs(demo["beta_hat"]).max())
    im0 = axes[0].imshow(demo["train"].beta_grid.T, origin="lower",
                         extent=[0, 1, 0, 1], cmap="RdBu_r",
                         vmin=-v_true, vmax=v_true, aspect="auto")
    axes[0].set_title(r"True $\beta(s, t)$")
    axes[0].set_xlabel("s"); axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(demo["beta_hat"].T, origin="lower",
                         extent=[0, 1, 0, 1], cmap="RdBu_r",
                         vmin=-v_hat, vmax=v_hat, aspect="auto")
    est = demo.get("estimator", "ols")
    if est == "ridge":
        sub = rf"ridge, $\lambda$={demo['lambda']:.2g}"
    else:
        sub = "OLS"
    axes[1].set_title(rf"$\hat\beta = \sqrt{{p}}\,\hat B$ ({sub}, p={p}, n={demo['n']})")
    axes[1].set_xlabel("s"); axes[1].set_ylabel("t")
    fig.colorbar(im1, ax=axes[1])
    _save_fig(fig, out_dir, "estimated_B_heatmap")
    plt.close(fig)


def plot_example_curves(demo: dict, out_dir: Path, n_curves: int = 4) -> None:
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(replication_seed(SWEEP_IDS["demo"], 1))
    idx = rng.choice(demo["test"].Y_noisy.shape[0], size=n_curves, replace=False)
    t = demo["test"].t_grid
    fig, axes = plt.subplots(1, n_curves, figsize=(3.2 * n_curves, 3.2),
                             sharey=True)
    if n_curves == 1:
        axes = [axes]
    for ax, k in zip(axes, idx):
        ax.plot(t, demo["test"].Y_noisy[k], color=NEUTRAL_GREY, lw=0.9,
                label=r"$Y^{\rm test}$ (noisy)")
        ax.plot(t, demo["test"].Y_signal[k], color=AUBURN_NAVY, lw=1.6,
                label=r"$T_\beta X^{\rm test}$")
        ax.plot(t, demo["Y_signal_hat"][k], color=AUBURN_ORANGE, lw=1.6,
                ls="--", label=r"$T_{\hat\beta} X^{\rm test}$")
        ax.set_xlabel("t")
        ax.set_title(f"test curve k={k}")
    axes[0].set_ylabel("response")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Example test curves at p={demo['p']}, n={demo['n']}")
    _save_fig(fig, out_dir, "example_true_vs_predicted_curves")
    plt.close(fig)


def plot_sweep_A(df_A: pd.DataFrame, out_dir: Path) -> dict:
    """Discretization error (primary) plus OpErr and sqrt(ISE) on log-log axes.

    Per spec section 7 (revised), DiscErr = ||T_beta^mu - T_beta^{mu_ref}||
    is what scales as p^-2 for smooth beta under midpoint discretization;
    OpErr and ISE are estimation-error metrics that are reported alongside
    for context but are not expected to follow the discretization rate when
    finite-sample variance dominates bias.
    """
    import matplotlib.pyplot as plt
    agg = aggregate(df_A, ["p"], metric_cols=("DiscErr", "OpErr", "ISE"))
    agg = agg.sort_values("p")
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.errorbar(agg["p"], agg["DiscErr_mean"], yerr=agg["DiscErr_se"],
                color=AUBURN_NAVY, marker="o", lw=1.8, label="DiscErr (primary)")
    ax.errorbar(agg["p"], agg["OpErr_mean"], yerr=agg["OpErr_se"],
                color=AUBURN_ORANGE, marker="s", lw=1.2, label="OpErr")
    ax.errorbar(agg["p"], np.sqrt(agg["ISE_mean"]), yerr=0.0,
                color=NEUTRAL_GREY, marker="^", lw=1.2,
                label=r"$\sqrt{\rm ISE}$")
    # Reference p^-2 guide-line anchored at the smallest p:
    p0 = agg["p"].values[0]
    ref0 = agg["DiscErr_mean"].values[0]
    p_line = agg["p"].values
    ax.plot(p_line, ref0 * (p_line / p0) ** -2, color=NEUTRAL_GREY,
            ls=":", lw=1.0, label=r"$p^{-2}$ reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("grid size p")
    ax.set_ylabel("error")
    ax.set_title("Sweep A: discretization gap vs grid density")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    lp = np.log(agg["p"].values)
    slope_DiscErr = float(np.polyfit(lp, np.log(agg["DiscErr_mean"].values), 1)[0])
    slope_OpErr = float(np.polyfit(lp, np.log(agg["OpErr_mean"].values), 1)[0])
    slope_ISE = float(np.polyfit(lp, np.log(agg["ISE_mean"].values), 1)[0])
    _save_fig(fig, out_dir, "operator_error_vs_grid_size")
    plt.close(fig)
    return {"slope_DiscErr": slope_DiscErr, "slope_OpErr": slope_OpErr,
            "slope_ISE": slope_ISE, "table": agg}


def plot_sweep_B(df_B: pd.DataFrame, out_dir: Path) -> dict:
    import matplotlib.pyplot as plt
    agg = aggregate(df_B, ["n"], metric_cols=("OpErr", "RMSE"))
    agg = agg.sort_values("n")
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.errorbar(agg["n"], agg["RMSE_mean"], yerr=agg["RMSE_se"],
                color=AUBURN_NAVY, marker="o", lw=1.5, label="RMSE")
    ax.errorbar(agg["n"], agg["OpErr_mean"], yerr=agg["OpErr_se"],
                color=AUBURN_ORANGE, marker="s", lw=1.5, label="OpErr")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sample size n (p=80)")
    ax.set_ylabel("error")
    ax.set_title("Sweep B: prediction error vs sample size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ln = np.log(agg["n"].values)
    slope_OpErr = float(np.polyfit(ln, np.log(agg["OpErr_mean"].values), 1)[0])
    slope_RMSE = float(np.polyfit(ln, np.log(agg["RMSE_mean"].values), 1)[0])
    _save_fig(fig, out_dir, "prediction_error_by_grid_size")
    plt.close(fig)
    return {"slope_OpErr": slope_OpErr, "slope_RMSE": slope_RMSE,
            "table": agg}


def plot_sweep_C(df_C: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    agg = aggregate(df_C, ["p", "n", "estimator"],
                    metric_cols=("OpErr", "RMSE", "kappa"))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for est, color, marker in (("ols", NEUTRAL_GREY, "o"),
                               ("ridge", AUBURN_NAVY, "s")):
        sub = agg[agg["estimator"] == est].sort_values("kappa_mean")
        ax.errorbar(sub["kappa_mean"], sub["OpErr_mean"],
                    yerr=sub["OpErr_se"], color=color, marker=marker,
                    lw=1.4, label=est.upper())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"condition number $\kappa(X^\top X)$")
    ax.set_ylabel("OpErr")
    ax.set_title("Sweep C: OLS vs ridge under varying conditioning")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save_fig(fig, out_dir, "ridge_vs_ols_conditioning")
    plt.close(fig)
    return agg


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_full_simulation(R_A: int, R_B: int, R_C: int,
                        output_dir: str | Path,
                        n_jobs: int = -1,
                        target_snr: float = 5.0,
                        K: int = K_DEFAULT,
                        verbose: bool = True) -> dict:
    """Top-level orchestrator: calibrate sigma, run all sweeps, save CSVs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cal = calibrate_sigma(target_snr=target_snr, K=K)
    sigma = cal["sigma"]
    with open(out / "calibration.json", "w") as fh:
        json.dump(cal, fh, indent=2)
    if verbose:
        print(f"[calibrate] sigma={sigma:.6f} (signal_var={cal['signal_var']:.6f}, SNR={target_snr})")

    timings = {}

    if verbose: print(f"[sweep A] R={R_A}")
    t0 = time.time()
    df_A = run_sweep_A(R=R_A, sigma=sigma, n_jobs=n_jobs)
    timings["A"] = time.time() - t0
    df_A.to_csv(out / "sweep_A.csv", index=False)

    if verbose: print(f"[sweep B] R={R_B}")
    t0 = time.time()
    df_B = run_sweep_B(R=R_B, sigma=sigma, n_jobs=n_jobs)
    timings["B"] = time.time() - t0
    df_B.to_csv(out / "sweep_B.csv", index=False)

    if verbose: print(f"[sweep C] R={R_C}")
    t0 = time.time()
    df_C = run_sweep_C(R=R_C, sigma=sigma, n_jobs=n_jobs)
    timings["C"] = time.time() - t0
    df_C.to_csv(out / "sweep_C.csv", index=False)

    if verbose:
        for k, v in timings.items():
            print(f"[timing] sweep {k}: {v:.1f}s")

    return {"sigma": sigma, "calibration": cal, "df_A": df_A, "df_B": df_B,
            "df_C": df_C, "timings": timings}
