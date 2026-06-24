#!/usr/bin/env python3
"""
Extract losses from a training transcript and plot loss vs. log(data samples).

Handles:
  - "Epoch N: avg loss X / took ... / current LR [...]"  -> a loss point
  - "<value> +/- <stderr>" eval lines (mse / accuracy)   -> ignored
  - epoch-counter resets between stitched sub-runs        -> we use a running
    cumulative sample count, not the (resetting) epoch number.

Sample bookkeeping:
  - File 1 (this transcript): each logged loss point == 256,000 samples.
    400 points -> 102.4M samples at the end.
  - Continuation run: batch size 4,096,000 samples per point.
    25 points -> +102.4M -> 204.8M samples at the end.
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

# ---------------------------------------------------------------------------
# 1. Extract losses from the transcript
# ---------------------------------------------------------------------------
# Match the "avg loss <number>" pattern only. Eval lines like
# "4.0919... +/- 0.0105..." have no "avg loss" and are skipped automatically.
LOSS_RE = re.compile(r"avg loss\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def extract_losses(path):
    losses = []
    with open(path) as f:
        for line in f:
            m = LOSS_RE.search(line)
            if m:
                losses.append(float(m.group(1)))
    return losses

# ---------------------------------------------------------------------------
# 2. Build the two parallel lists: samples seen, and loss
# ---------------------------------------------------------------------------
SAMPLES_PER_POINT_FILE1 = 256_000      # samples per logged point in file 1
# NOTE: this is samples per *logged point*, not the batch size. The continuation
# runs log every 1000 steps at a true batch size of 4096, so each logged point
# covers 1000 * 4096 = 4,096,000 samples. (The "4,096,000" was originally a
# mislabel of epochs/batches.)
CONT_SAMPLES_PER_POINT  = 1000 * 4096  # = 4,096,000 samples per logged point

# Continuation runs, in order. Each is a separate run pasted in by hand, with its
# own per-point batch size. They keep counting samples from where file 1 ends.
# Run 1: LR ramp-down at 5e-07..; Run 2: LR 2e-07. Both batch 4,096,000.
continuation_runs = [
    {
        "samples_per_point": CONT_SAMPLES_PER_POINT,
        "losses": [
            0.13072545379400252, 0.13087643533945084, 0.13093588948249818,
            0.13099549114704132, 0.13145092725753785, 0.13076873421669005,
            0.12926418483257293, 0.1307376131415367, 0.13153896778821944,
            0.13146672248840333, 0.12997920215129852, 0.1308931827545166,
            0.13123989552259446, 0.1293485641479492, 0.12991385981440545,
            0.1312054380774498, 0.13055190145969392, 0.13073142766952514,
            0.13056129366159439, 0.1292877197265625, 0.13071723133325577,
            0.12993521839380265, 0.12922002673149108, 0.13022714406251906,
            0.12980404645204544,
        ],
    },
    {
        "samples_per_point": CONT_SAMPLES_PER_POINT,
        "losses": [
            0.13091165721416473, 0.1303046628832817, 0.13012198209762574,
            0.12973152250051498, 0.13066169768571853, 0.12966708540916444,
            0.13138784021139144, 0.12978260219097137, 0.12908771857619286,
            0.13012869656085968, 0.12949378788471222, 0.13030917346477508,
            0.13040849417448044, 0.13014354556798935, 0.12960705608129502,
            0.1298136979341507, 0.12958032935857772, 0.12959981709718704,
            0.1295458048582077, 0.131245157122612, 0.13007218837738038,
            0.130695378780365, 0.13042979091405868, 0.13027773648500443,
            0.1298159345984459,
        ],
    },
]

def build_series(transcript_path):
    file1_losses = extract_losses(transcript_path)

    samples = []
    losses  = []
    boundaries = []  # cumulative-sample positions where each continuation begins

    # File 1: cumulative count grows 256k per point regardless of epoch resets.
    cum = 0
    for loss in file1_losses:
        cum += SAMPLES_PER_POINT_FILE1
        samples.append(cum)
        losses.append(loss)
    n_file1 = len(file1_losses)

    # Each continuation keeps counting from where the previous segment ended.
    for run in continuation_runs:
        boundaries.append(cum)
        for loss in run["losses"]:
            cum += run["samples_per_point"]
            samples.append(cum)
            losses.append(loss)

    return (np.array(samples, dtype=float),
            np.array(losses, dtype=float),
            n_file1, boundaries)

# ---------------------------------------------------------------------------
# 3. Centered sliding-window moving average for smoothing
# ---------------------------------------------------------------------------
def moving_average(values, window=21):
    """Centered rolling mean. Near the edges the window shrinks to whatever
    points are available, so the smoothed line stays centered on real data
    from the very first point (no startup lag like an EMA has)."""
    n = len(values)
    half = window // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = values[lo:hi].mean()
    return out

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
def main(transcript_path="transcript.txt", out_path="training_loss.png", window=21):
    samples, losses, n_file1, boundaries = build_series(transcript_path)
    smooth = moving_average(losses, window=window)

    print(f"file 1 points : {n_file1}  -> {n_file1 * SAMPLES_PER_POINT_FILE1:,} samples")
    for k, run in enumerate(continuation_runs, 1):
        npts = len(run["losses"])
        print(f"continuation {k}: {npts}  -> {npts * run['samples_per_point']:,} samples")
    print(f"total points  : {len(losses)}")
    print(f"total samples : {int(samples[-1]):,}")

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Excess loss over an assumed asymptotic floor. Values <= floor have no log,
    # so mask them to NaN (they simply won't be drawn).
    FLOOR = 0.13
    def excess(arr):
        e = arr - FLOOR
        e[e <= 0] = np.nan
        return e
    raw_excess    = excess(losses.copy())
    smooth_excess = excess(smooth.copy())
    n_dropped = int(np.isnan(raw_excess).sum())

    ax.plot(samples, raw_excess, color="#bcbcbc", lw=1.0, alpha=0.8,
            label="loss − 0.13 (raw)", zorder=1)
    ax.plot(samples, smooth_excess, color="#1f5fbf", lw=2.2,
            label=f"loss − 0.13 ({window}-point sliding window)", zorder=3)

    # mark where the large-batch phase begins (first continuation only)
    if boundaries:
        y_top = np.nanmax(raw_excess)
        ax.axvline(boundaries[0], color="#7f8c8d", ls="--", lw=1.0, alpha=0.7, zorder=2)
        ax.annotate("continuation runs begin\n(batch size 4,096)",
                    xy=(boundaries[0], y_top),
                    xytext=(boundaries[0] * 0.97, y_top),
                    fontsize=8, color="#7f8c8d", va="top", ha="right")

    ax.set_xscale("linear")
    ax.set_yscale("log")

    # Plain numbers (with thousands separators) instead of 1e8-style labels.
    plain = FuncFormatter(lambda x, _: f"{int(x):,}")
    ax.xaxis.set_major_formatter(plain)
    ax.tick_params(axis="x", which="major", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlim(left=0)

    # loss-0.13 is log-scaled; let matplotlib pick decade ticks, shown plainly.
    loss_plain = FuncFormatter(lambda y, _: f"{y:g}")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(loss_plain)
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5)))
    ax.yaxis.set_minor_formatter(loss_plain)
    ax.tick_params(axis="y", which="major", labelsize=8)
    ax.tick_params(axis="y", which="minor", labelsize=7)

    ax.set_xlabel("Data samples seen")
    ax.set_ylabel("loss − 0.13 (log scale)")
    ax.set_title("Training loss vs. data samples seen")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved plot -> {out_path}")

if __name__ == "__main__":
    import sys
    tp = sys.argv[1] if len(sys.argv) > 1 else "transcript.txt"
    main(tp, "/mnt/user-data/outputs/training_loss.png")