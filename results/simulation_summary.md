# Simulation summary — unified operator framework

`σ` calibrated to SNR = 5 from a pilot of (n=1000, p=200): **σ = 0.193587** (signal_var = 0.187380).

Across all three sweeps, OLS and ridge are two estimators of the *same* discrete-measure operator `T_β^μ`, not two competing regression models. The grid-size results characterize the gap between `T_β^μ` and `T_β` as the discretization `μ` becomes finer — measured by `DiscErr` against a fine reference grid `μ_ref` (`p_ref` = 2000). The sample-size results characterize the standard estimation-error scaling at fixed `μ`. The conditioning results characterize how spectral regularization stabilizes the same Gram operator under poor conditioning.

## Sweep A — grid density (n=500, OLS)

Log-log slope of **DiscErr** vs p: **-2.948** (full range, R=100/cell). Asymptotic slope (p ≥ 40, after the small-p transient): **-2.025** — the Riemann-rule rate `p^-2`.

Log-log slope of OpErr vs p: **+0.240** (estimation-error metric at fixed μ; not expected to track the discretization rate).

Log-log slope of ISE vs p:   **+1.502**.

|   p |   DiscErr_mean |   DiscErr_se |   OpErr_mean |   OpErr_se |   ISE_mean |    ISE_se |   RMSE_mean |   RMSE_se |   kappa_mean |   kappa_se |   R |
|----:|---------------:|-------------:|-------------:|-----------:|-----------:|----------:|------------:|----------:|-------------:|-----------:|----:|
|  10 |      0.04693   |    0.0001175 |      0.02774 |  0.0001702 |    0.01903 | 0.0002774 |      0.1956 | 0.0001857 |   52.59      |  0.4534    | 100 |
|  20 |      0.02111   |    4.768e-05 |      0.03954 |  0.0001404 |    0.161   | 0.001371  |      0.1975 | 0.0001414 |  253.1       |  2.325     | 100 |
|  40 |      0.0002166 |    6.809e-07 |      0.05733 |  0.0001228 |    1.412   | 0.00716   |      0.2017 | 9.749e-05 | 1212         |  9.211     | 100 |
|  80 |      5.28e-05  |    1.604e-07 |      0.06476 |  8.889e-05 |    3.601   | 0.01208   |      0.2041 | 7.532e-05 |    9.507e+18 |  1.484e+18 | 100 |
| 160 |      1.311e-05 |    4.195e-08 |      0.06463 |  6.171e-05 |    3.59    | 0.008932  |      0.2041 | 5.401e-05 |    2.449e+19 |  4.893e+18 | 100 |
| 320 |      3.2e-06   |    1.092e-08 |      0.06465 |  5.401e-05 |    3.59    | 0.007221  |      0.2041 | 4.163e-05 |    1.389e+20 |  3.639e+19 | 100 |

## Sweep B — sample size (p=80, OLS)

Log-log slope of OpErr vs n: **-0.608** (R=100/cell, expected ≈ -0.5).

Log-log slope of RMSE  vs n: **-0.106** (RMSE flattens at the noise floor σ).

|    n |   OpErr_mean |   OpErr_se |   RMSE_mean |   RMSE_se |   ISE_mean |   ISE_se |   kappa_mean |   kappa_se |   R |
|-----:|-------------:|-----------:|------------:|----------:|-----------:|---------:|-------------:|-----------:|----:|
|  100 |      0.1952  |  0.0005883 |      0.2749 | 0.0004328 |    32.88   | 0.2131   |    8.048e+18 |  1.505e+18 | 100 |
|  250 |      0.0969  |  0.0001638 |      0.2164 | 0.0001135 |     8.079  | 0.03345  |    1.566e+19 |  5.881e+18 | 100 |
|  500 |      0.06456 |  9.861e-05 |      0.204  | 7.965e-05 |     3.577  | 0.01232  |    3.194e+19 |  2.153e+19 | 100 |
| 1000 |      0.04436 |  5.733e-05 |      0.1987 | 6.876e-05 |     1.691  | 0.005369 |    1.407e+19 |  3.073e+18 | 100 |
| 2000 |      0.03104 |  3.65e-05  |      0.1962 | 7.799e-05 |     0.8247 | 0.002499 |    1.36e+19  |  2.448e+18 | 100 |

## Sweep C — conditioning (OLS vs ridge)

R=50/cell. Ridge λ selected by 5-fold CV on a 25-point log grid in [10⁻⁶, 10²]; one SVD of `X_mat` per fold and one for the final fit.

|   p |   n | estimator   |   OpErr_mean |   OpErr_se |   RMSE_mean |   RMSE_se |   kappa_mean |    kappa_se |   R |
|----:|----:|:------------|-------------:|-----------:|------------:|----------:|-------------:|------------:|----:|
|  40 |  20 | ols         |      0.2565  |  0.002368  |      0.3213 | 0.001898  |    7.453e+18 |   1.83e+18  |  50 |
|  40 |  20 | ridge       |      0.09809 |  0.001415  |      0.2171 | 0.000659  |    7.453e+18 |   1.83e+18  |  50 |
|  40 |  40 | ols         |      4.495   |  0.8857    |      4.503  | 0.8853    |    3.558e+07 |   1.838e+07 |  50 |
|  40 |  40 | ridge       |      0.0741  |  0.0008569 |      0.2072 | 0.0003545 |    3.558e+07 |   1.838e+07 |  50 |
|  40 |  80 | ols         |      0.1949  |  0.001089  |      0.2747 | 0.0007686 | 5005         | 174.7       |  50 |
|  40 |  80 | ridge       |      0.05607 |  0.0005198 |      0.2015 | 0.0001778 | 5005         | 174.7       |  50 |
|  80 |  40 | ols         |      0.4404  |  0.005059  |      0.4813 | 0.004637  |    1.658e+19 |   3.834e+18 |  50 |
|  80 |  40 | ridge       |      0.07357 |  0.0007687 |      0.207  | 0.0002952 |    1.658e+19 |   3.834e+18 |  50 |
|  80 |  80 | ols         |      0.2509  |  0.00136   |      0.3168 | 0.00108   |    1.259e+19 |   5.03e+18  |  50 |
|  80 |  80 | ridge       |      0.05561 |  0.0005115 |      0.2013 | 0.0001731 |    1.259e+19 |   5.03e+18  |  50 |
|  80 | 160 | ols         |      0.1312  |  0.000375  |      0.2338 | 0.0002297 |    3.047e+19 |   1.444e+19 |  50 |
|  80 | 160 | ridge       |      0.04177 |  0.0002022 |      0.1979 | 0.0001161 |    3.047e+19 |   1.444e+19 |  50 |
| 160 |  80 | ols         |      0.2518  |  0.001193  |      0.3176 | 0.0009662 |    4.927e+19 |   1.64e+19  |  50 |
| 160 |  80 | ridge       |      0.05571 |  0.0004526 |      0.2014 | 0.0001562 |    4.927e+19 |   1.64e+19  |  50 |
| 160 | 160 | ols         |      0.1308  |  0.000328  |      0.2336 | 0.0002192 |    1.779e+19 |   3.297e+18 |  50 |
| 160 | 160 | ridge       |      0.04207 |  0.0002274 |      0.1981 | 9.533e-05 |    1.779e+19 |   3.297e+18 |  50 |
| 160 | 320 | ols         |      0.08365 |  0.0001653 |      0.2109 | 9.647e-05 |    2.068e+19 |   4.42e+18  |  50 |
| 160 | 320 | ridge       |      0.03156 |  0.0001514 |      0.1962 | 7.446e-05 |    2.068e+19 |   4.42e+18  |  50 |
| 320 | 160 | ols         |      0.131   |  0.0002889 |      0.2337 | 0.0001685 |    4.667e+20 |   3.84e+20  |  50 |
| 320 | 160 | ridge       |      0.04178 |  0.0001753 |      0.198  | 6.128e-05 |    4.667e+20 |   3.84e+20  |  50 |
| 320 | 320 | ols         |      0.08373 |  0.0001202 |      0.2109 | 7.017e-05 |    8.775e+19 |   2.886e+19 |  50 |
| 320 | 320 | ridge       |      0.03192 |  0.0001671 |      0.1962 | 5.17e-05  |    8.775e+19 |   2.886e+19 |  50 |
| 320 | 640 | ols         |      0.05652 |  6.671e-05 |      0.2017 | 5.223e-05 |    9.636e+19 |   2.145e+19 |  50 |
| 320 | 640 | ridge       |      0.02432 |  6.454e-05 |      0.1951 | 4.571e-05 |    9.636e+19 |   2.145e+19 |  50 |

## Heatmap demo

`estimated_B_heatmap.png` shows the ridge β_hat at (n=500, p=80) with CV-selected λ = 4.642. corr(β_hat, β_true) = 0.920; mean |β_hat − β| = 0.178.


## Internal consistency

- Sweep A: mean(RMSE²)=0.040503, mean(OpErr² + σ²)=0.040506, rel diff = 0.0070%
- Sweep B: mean(RMSE²)=0.048403, mean(OpErr² + σ²)=0.048405, rel diff = 0.0041%
- Sweep C: mean(RMSE²)=2.503061, mean(OpErr² + σ²)=2.502919, rel diff = 0.0057%