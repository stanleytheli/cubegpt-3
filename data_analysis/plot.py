#!/usr/bin/env python3
"""
Power-law view with a logarithmic smoothing window and a fitted floor L_inf.

Model:   loss(steps) = L_inf + C * steps^(-a)
A clean power law is a straight line on log(loss - L_inf) vs log(steps).

Smoothing: logarithmic (multiplicative) window. For each point at x = steps[i],
the smoothed value is the arithmetic mean of every point whose x lies in
[x / FACTOR, x * FACTOR]. With FACTOR = 1.5 that is a window of constant width
in log-space, which matches the log x-axis (unlike a fixed point-count window).

Steps: the log's epoch counter increments by 1000 per logged line and its
per-segment maxima match 1000 steps per logged point, so each logged point is
treated as 1000 training steps (ending at 450k).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
from scipy.optimize import curve_fit

from plot_training import build_series

STEPS_PER_POINT = 1000
FACTOR          = 2.0         # logarithmic smoothing window: x/1.5 .. x*1.5
X_MODE          = "samples"   # "samples" or "steps"


def log_window_mean(x, y, factor=1.5):
    """Arithmetic mean of all y whose x is within [x_i/factor, x_i*factor]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    for i, xi in enumerate(x):
        lo, hi = xi / factor, xi * factor
        mask = (x >= lo) & (x <= hi)
        out[i] = y[mask].mean()
    return out


def powerlaw(s, L_inf, C, a):
    return L_inf + C * np.power(s, -a)


def fit_powerlaw(x, values):
    # L_inf bounded above the noise floor (so the asymptote can sit near ~0.13);
    # the log-plot positivity is handled separately by masking. C bounds are wide
    # because its scale depends on the x units (steps vs samples).
    p0     = [0.128, 1.0, 0.4]
    bounds = ([0.05, 1e-12, 0.01], [0.135, 1e12, 3.0])
    popt, _ = curve_fit(powerlaw, x, values, p0=p0, bounds=bounds, maxfev=100000)
    return popt  # L_inf, C, a


def main(transcript_path="data_analysis/transcript.txt", out_path=None):
    samples, losses, n_file1, boundaries = build_series(transcript_path)
    steps = np.arange(1, len(losses) + 1) * STEPS_PER_POINT

    if X_MODE == "samples":
        x        = samples
        xlabel   = "Data samples seen"
        boundary = boundaries[0]                       # 102.4M samples
        suffix   = "samples"
    else:
        x        = steps
        xlabel   = "Training steps"
        boundary = n_file1 * STEPS_PER_POINT           # 400k steps
        suffix   = "steps"
    if out_path is None:
        out_path = f"data_analysis/output/out_sam_f20.png"

    smooth = log_window_mean(x, losses, factor=FACTOR)

    # Fit the floor (and the power law) on the log-window-smoothed curve.
    L_inf, C, a = fit_powerlaw(x, smooth)
    print(f"x-axis       : {xlabel}")
    print(f"fitted L_inf : {L_inf:.6f}")
    print(f"fitted C     : {C:.6g}")
    print(f"fitted a     : {a:.4f}   (slope on log-log)")

    resid_raw    = losses - L_inf
    resid_smooth = smooth - L_inf
    fit_resid    = C * np.power(x, -a)   # straight line on log-log

    n_neg = int((resid_raw <= 0).sum())
    print(f"raw points <= L_inf : {n_neg} / {len(losses)} (dropped from log plot)")
    print(f"x range      : {int(x[0]):,} .. {int(x[-1]):,}")

    raw_pos = np.where(resid_raw > 0, resid_raw, np.nan)
    sm_pos  = np.where(resid_smooth > 0, resid_smooth, np.nan)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(x, raw_pos, color="#cfcfcf", lw=1.0, alpha=0.8,
            label=f"loss \u2212 L\u221e (raw, where > 0)", zorder=1)
    ax.plot(x, sm_pos, color="#1f5fbf", lw=2.2,
            label=f"loss \u2212 L\u221e (log window \u00d7{FACTOR})", zorder=3)
    ax.plot(x, fit_resid, color="#c0392b", lw=1.6, ls="--",
            label=f"power-law fit: C\u00b7{suffix}^(\u2212{a:.3f})", zorder=4)

    ax.axvline(boundary, color="#7f8c8d", ls="--", lw=1.0, alpha=0.7, zorder=2)
    ax.annotate("continuation runs begin\n(batch size 4,096)",
                xy=(boundary, np.nanmax(sm_pos)),
                xytext=(boundary * 0.97, np.nanmax(sm_pos)),
                fontsize=8, color="#7f8c8d", va="top", ha="right")

    ax.set_xscale("log")
    ax.set_yscale("log")

    plain = FuncFormatter(lambda v, _: f"{int(v):,}")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    ax.xaxis.set_minor_formatter(plain)
    plt.setp(ax.get_xticklabels(which="both"), rotation=30, ha="right", fontsize=8)

    gfmt = FuncFormatter(lambda y, _: f"{y:g}")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(gfmt)
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    ax.yaxis.set_minor_formatter(gfmt)
    ax.tick_params(axis="y", which="major", labelsize=8)
    ax.tick_params(axis="y", which="minor", labelsize=7)
    ax.set_ylim(4e-4, 5e-2)

    ax.set_xlabel(f"{xlabel} (log scale)")
    ax.set_ylabel("loss \u2212 L\u221e  (log scale)")
    ax.set_title(f"Residual loss vs. {xlabel.lower()}  (log\u2013log, fitted L\u221e = {L_inf:.4f})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved plot -> {out_path}")


if __name__ == "__main__":
    import sys
    tp = sys.argv[1] if len(sys.argv) > 1 else "data_analysis/transcript.txt"
    main(tp)