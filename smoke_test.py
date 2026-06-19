"""Smoke test for simulation_utils.

Runs all three sweeps with R=5, generates figures, and reports diagnostics:
  - file structure under results/
  - qualitative patterns (operator error decreasing with p, RMSE with n,
    ridge stabilizing OLS in Sweep C)
  - internal consistency RMSE^2 ~ OpErr^2 + sigma^2
  - heatmap visual sanity check (numerical correlation of beta_hat vs beta)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import simulation_utils as su


def section(msg: str) -> None:
    print(f"\n{'=' * 64}\n{msg}\n{'=' * 64}")


def main() -> int:
    R = 5
    out = Path("results")
    n_jobs = 4

    section(f"Running smoke test with R={R} across sweeps A, B, C")
    res = su.run_full_simulation(R_A=R, R_B=R, R_C=R, output_dir=out,
                                 n_jobs=n_jobs, verbose=True)
    sigma = res["sigma"]

    section("Sweep A aggregate (DiscErr primary; OpErr, ISE secondary; OLS, n=500)")
    agg_A = su.aggregate(res["df_A"], ["p"],
                         metric_cols=("DiscErr", "OpErr", "ISE", "RMSE", "kappa"))
    agg_A = agg_A.sort_values("p").reset_index(drop=True)
    print(agg_A.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    lp = np.log(agg_A["p"].values)
    slope_DiscErr = np.polyfit(lp, np.log(agg_A["DiscErr_mean"].values), 1)[0]
    slope_OpErr = np.polyfit(lp, np.log(agg_A["OpErr_mean"].values), 1)[0]
    slope_ISE = np.polyfit(lp, np.log(agg_A["ISE_mean"].values), 1)[0]
    print(f"  log-log slope DiscErr vs p = {slope_DiscErr:+.3f}  (expected ~ -2)")
    print(f"  log-log slope OpErr   vs p = {slope_OpErr:+.3f}  (estimation-error term, no expected rate)")
    print(f"  log-log slope ISE     vs p = {slope_ISE:+.3f}")

    section("Sweep B aggregate (OpErr, RMSE vs n; OLS, p=80)")
    agg_B = su.aggregate(res["df_B"], ["n"],
                         metric_cols=("OpErr", "RMSE", "ISE", "kappa"))
    agg_B = agg_B.sort_values("n").reset_index(drop=True)
    print(agg_B.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    section("Sweep C aggregate (OLS vs Ridge under varying conditioning)")
    agg_C = su.aggregate(res["df_C"], ["p", "n", "estimator"],
                         metric_cols=("OpErr", "RMSE", "kappa"))
    agg_C = agg_C.sort_values(["p", "n", "estimator"]).reset_index(drop=True)
    print(agg_C.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ------------------------------------------------------------------
    # Internal consistency check: RMSE^2 ~ OpErr^2 + sigma^2
    # ------------------------------------------------------------------
    section("Internal consistency: RMSE^2 vs OpErr^2 + sigma^2")
    for label, df in [("A", res["df_A"]), ("B", res["df_B"]),
                      ("C", res["df_C"])]:
        df = df.copy()
        df["lhs"] = df["RMSE"] ** 2
        df["rhs"] = df["OpErr"] ** 2 + sigma ** 2
        df["abs_err"] = (df["lhs"] - df["rhs"]).abs()
        df["rel_err"] = df["abs_err"] / df["rhs"]
        print(f"  Sweep {label}: mean(LHS)={df['lhs'].mean():.6f}, "
              f"mean(RHS)={df['rhs'].mean():.6f}, "
              f"mean|LHS-RHS|={df['abs_err'].mean():.6f}, "
              f"mean rel err={df['rel_err'].mean():.4f}")
    print(f"  sigma^2 = {sigma**2:.6f}")

    # ------------------------------------------------------------------
    # Qualitative pattern checks
    # ------------------------------------------------------------------
    section("Qualitative patterns")
    # A: OpErr should decrease monotonically (or close to) with p
    a = agg_A.sort_values("p")
    print(f"  A: OpErr by p: " +
          ", ".join(f"p={int(p)}->{m:.4f}"
                    for p, m in zip(a["p"], a["OpErr_mean"])))
    a_ratio = a["OpErr_mean"].values[-1] / a["OpErr_mean"].values[0]
    print(f"     OpErr ratio (largest p / smallest p) = {a_ratio:.4f}  "
          f"[expected << 1; spec predicts ~p^-2 trend, but smoke R=5 is noisy]")

    # B: RMSE should decrease with n until floored
    b = agg_B.sort_values("n")
    print(f"  B: RMSE by n: " +
          ", ".join(f"n={int(n)}->{m:.4f}"
                    for n, m in zip(b["n"], b["RMSE_mean"])))
    b_ratio = b["RMSE_mean"].values[-1] / b["RMSE_mean"].values[0]
    print(f"     RMSE ratio (largest n / smallest n) = {b_ratio:.4f}")

    # C: ridge should beat OLS at low n/p (high kappa)
    print("  C: per-cell OLS-vs-Ridge OpErr:")
    pivot = (agg_C.pivot_table(index=["p", "n"], columns="estimator",
                               values="OpErr_mean")
             .reset_index())
    pivot["ratio_ols_over_ridge"] = pivot["ols"] / pivot["ridge"]
    print(pivot.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ------------------------------------------------------------------
    # Demo replication: figures + heatmap correlation
    # ------------------------------------------------------------------
    section("Demo replication for figures (p=80, n=500, default = ridge)")
    demo = su.demo_replication(sigma=sigma, n=500, p=80)
    print(f"  estimator: {demo['estimator']}, lambda: {demo['lambda']:.4g}")
    print(f"  metrics: {demo['metrics']}")
    flat_corr = float(np.corrcoef(demo["beta_hat"].ravel(),
                                  demo["train"].beta_grid.ravel())[0, 1])
    abs_diff = float(np.abs(demo["beta_hat"] - demo["train"].beta_grid).mean())
    print(f"  corr(beta_hat, beta_true) = {flat_corr:.4f}, "
          f"mean |beta_hat - beta| = {abs_diff:.4f}")

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    section("Generating figures under results/figures/")
    fig_dir = out / "figures"
    su.plot_true_beta_surface(fig_dir)
    su.plot_estimated_B_heatmap(demo, fig_dir)
    su.plot_example_curves(demo, fig_dir)
    su.plot_sweep_A(res["df_A"], fig_dir)
    su.plot_sweep_B(res["df_B"], fig_dir)
    su.plot_sweep_C(res["df_C"], fig_dir)

    # ------------------------------------------------------------------
    # File structure check
    # ------------------------------------------------------------------
    section("File structure under results/")
    for path in sorted(out.rglob("*")):
        if path.is_file():
            sz = path.stat().st_size
            print(f"  {path.as_posix()}  ({sz} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
