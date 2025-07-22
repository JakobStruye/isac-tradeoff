import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from itertools import cycle
from collections import defaultdict
from math import ceil
from matplotlib.ticker import FixedLocator, FixedFormatter

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Plot test accuracy from log files.")
parser.add_argument("logdir", nargs="?", default="./logs", help="Primary log directory.")
parser.add_argument("--logdir2", help="Optional second log directory to compare (plotted transparently and dashed).")
parser.add_argument("--show_interleaved", action="store_true", help="Show interleaved results in plot.")
parser.add_argument("--actual_s_vals", action="store_true", help="Use actual S values based on effective rate (d / ceil(d/x)).")
parser.add_argument("--mean_std", action="store_true", help="Plot mean with ± std deviation instead of median with IQR.")
parser.add_argument("--savefig", action="store_true", help="Save the plot instead of displaying it.")  # NEW
args = parser.parse_args()

LOG_DIRS = [(args.logdir, 1.0, '-', 'black')]
if args.logdir2:
    LOG_DIRS.append((args.logdir2, 0.5, '--', 'gray'))

SHOW_INTERLEAVED = args.show_interleaved
USE_ACTUAL_S_VALS = args.actual_s_vals
USE_MEAN_STD = args.mean_std

# === Filename Pattern ===
pattern = re.compile(
    r"epoch(?P<epoch>\d+)"
    r"(?:_subtime(?P<subtime>\d+))?"
    r"(?:_subtx(?P<subtx>\d+)(?:_(?P<approach_tx>repeated|interleaved))?)?"
    r"(?:_subrx(?P<subrx>\d+)(?:_(?P<approach_rx>repeated|interleaved))?)?"
    r"(?:_(?P<approach_global>repeated|interleaved))?"
    r"_seed(?P<seed>\d+)\.log$"
)

def extract_approach(m):
    approaches = []
    for field in ["approach_tx", "approach_rx", "approach_global"]:
        if m.group(field):
            approaches.append(m.group(field))
    approaches = list(set(approaches))
    if len(approaches) > 1:
        raise ValueError(f"Conflicting approaches in filename: {m.string}")
    return approaches[0] if approaches else "single"

def extract_accuracy(log_path):
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None
        m = re.search(r"Test accuracy: ([0-9.]+)", lines[-2])
        if m:
            return float(m.group(1))
    except:
        pass
    return None

def load_results(log_dir):
    results = {}
    for fname in sorted(os.listdir(log_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        try:
            approach = extract_approach(m)
        except ValueError as e:
            print(e)
            continue
        params = m.groupdict()
        epoch = int(params["epoch"])
        seed = int(params["seed"])
        subtime = int(params["subtime"] or 1)
        subtx = int(params["subtx"] or 1)
        subrx = int(params["subrx"] or 1)
        path = os.path.join(log_dir, fname)
        acc = extract_accuracy(path)
        if acc is None:
            continue
        key = (epoch, subtime, subtx, subrx, approach)
        results.setdefault(key, []).append(acc)
    return results

def get_baseline(results, epoch):
    accs = []
    for (ep, subtime, subtx, subrx, _), vals in results.items():
        if ep == epoch and subtime == subtx == subrx == 1:
            accs.extend(vals)
    if accs:
        if USE_MEAN_STD:
            mean = np.mean(accs)
            std = np.std(accs)
            return mean, (std, std)  # symmetric error bars
        else:
            median = np.median(accs)
            q1 = np.percentile(accs, 25)
            q3 = np.percentile(accs, 75)
            return median, (median - q1, q3 - median)
    return None, (None, None)

def adjust_x(subsample, x_vals):
    def eff_rate(d, x):
        return d / np.ceil(d / x)
    x_vals = np.array(x_vals, dtype=float)
    if subsample == "subtime":
        return eff_rate(20, x_vals)
    elif subsample == "subtx":
        return eff_rate(50, x_vals)
    elif subsample == "subrx":
        return eff_rate(56, x_vals)
    elif subsample == "all_beams":
        tx = eff_rate(50, x_vals)
        rx = eff_rate(56, x_vals)
        return (0.5 * (tx + rx)) ** 2
    return x_vals

def collect_line_with_baseline(results, epoch, approach, subsample):
    baseline_val, _ = get_baseline(results, epoch)
    if baseline_val is None:
        return np.array([]), np.array([]), np.array([])

    raw_data = [(1, baseline_val)]
    for (ep, subtime, subtx, subrx, app), acc_list in results.items():
        if ep != epoch or app != approach:
            continue
        if subtime == 1 and subtx == 1 and subrx == 1:
            continue

        if subsample == "subtime" and app == "single" and subtx == 1 and subrx == 1:
            x = subtime
        elif subsample == "subtx" and subtime == 1 and subrx == 1:
            x = subtx
        elif subsample == "subrx" and subtime == 1 and subtx == 1:
            x = subrx
        elif subsample == "all_beams" and subtime == 1 and subtx == subrx:
            x = subtx
        else:
            continue

        for acc in acc_list:
            raw_data.append((x, acc))

    x_vals, accs = zip(*raw_data)
    x_vals = np.array(x_vals)
    accs = np.array(accs)

    if USE_ACTUAL_S_VALS:
        x_vals = adjust_x(subsample, x_vals)
    elif subsample == "all_beams":
        x_vals = x_vals ** 2

    grouped = defaultdict(list)
    for x, acc in zip(x_vals, accs):
        grouped[round(x, 6)].append(acc)

    x = sorted(grouped)
    if USE_MEAN_STD:
        m = np.array([np.mean(grouped[val]) for val in x])
        std = np.array([np.std(grouped[val]) for val in x])
        errs = np.vstack((std, std))  # symmetric error bars
    else:
        m = np.array([np.median(grouped[val]) for val in x])
        q1 = np.array([np.percentile(grouped[val], 25) for val in x])
        q3 = np.array([np.percentile(grouped[val], 75) for val in x])
        err_lower = m - q1
        err_upper = q3 - m
        errs = np.vstack((err_lower, err_upper))
    x = np.array(x)
    return x, m, errs

# === Plot setup ===
subsample_types = ["subtime", "subtx", "subrx", "all_beams"]
subsample_colors = {
    "subtime": "tab:blue",
    "subtx": "tab:orange",
    "subrx": "tab:green",
    "all_beams": "tab:red"
}
subsample_labels = {
    "subtime": "time",
    "subtx": "tx",
    "subrx": "rx",
    "all_beams": "tx+rx"
}
markers = cycle(['o', 's', '^', 'v', '<', '>', 'P', 'X', 'D', '*'])
offset_spacing = 0.15

fig, ax = plt.subplots(figsize=(12, 4))
ax2 = None
if USE_ACTUAL_S_VALS:
    ax2 = ax.twinx()
    x_line = np.array(np.arange(1,9.05,0.1))
    y_airtime = 1 - (1/x_line)
    ax2.plot(x_line, y_airtime, color='gray', linestyle=':', linewidth=2)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Communications airtime fraction", color='gray')
    ax2.tick_params(axis='y', colors='gray')

for dir_idx, (log_dir, alpha, linestyle, star_color) in reversed(list(enumerate(LOG_DIRS))):
    results = load_results(log_dir)
    epochs = sorted(set(k[0] for k in results))
    added_labels = set()

    for epoch in epochs:
        baseline_val, (err_low, err_high) = get_baseline(results, epoch)
        if baseline_val is None:
            continue

        yerr = np.array([[err_low], [err_high]]) if alpha == 1.0 else None
        ax.errorbar([1], [baseline_val],
                    yerr=yerr,
                    marker='*', color=star_color, markersize=10,
                    capsize=4 if alpha == 1.0 else 0,
                    linestyle='None',
                    label='baseline' if (alpha == 1.0 and "baseline" not in added_labels) else None,
                    zorder=10, alpha=alpha)
        if alpha == 1.0:
            added_labels.add("baseline")

        for subsample_idx, subsample in reversed(list(enumerate(subsample_types))):
            approaches = ["single"] if subsample == "subtime" else ["repeated", "interleaved"]
            for approach in approaches:
                if approach == "interleaved" and not SHOW_INTERLEAVED:
                    continue
                x, m, errs = collect_line_with_baseline(results, epoch, approach, subsample)
                if len(x) == 0:
                    continue

                label = subsample_labels[subsample] if (subsample_labels[subsample] not in added_labels and alpha == 1.0) else None
                color = subsample_colors[subsample]
                xplot = x.copy()

                if alpha == 1.0 and not USE_ACTUAL_S_VALS:
                    offset = (subsample_idx - 1.5) * (offset_spacing / 2)
                    xplot = np.where(x != 1, xplot + offset, xplot)
                print(subsample, xplot, m)
                ax.plot(xplot, m, linestyle=linestyle, color=color, alpha=alpha)

                if alpha == 1.0:
                    mask = (x != 1)
                    ax.scatter(xplot[mask], m[mask],
                               marker=next(markers),
                               color=color, alpha=alpha,
                               label=label if mask.any() else None,
                               zorder=3)
                    ax.errorbar(xplot[mask], m[mask], yerr=errs[:, mask],
                                fmt='none', ecolor=color,
                                capsize=0, alpha=alpha,
                                zorder=2)
                else:
                    ax.scatter(xplot, m, color=color, alpha=alpha, marker='.', zorder=1)

                if label:
                    added_labels.add(label)

ax.set_xlim(0.5, 9.5)
ax.set_ylim(90, 94.5)
ax.set_xlabel("Subsampling factor (target)" if not USE_ACTUAL_S_VALS else "Subsampling factor (actual)")
ax.set_ylabel("Accuracy")
ax.grid(True)

if USE_ACTUAL_S_VALS:
    main_ticks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    sensing_labels = ['1.00', '0.500', '0.333', '0.250', '0.200', '0.166', '0.143', '0.125', '0.111']
    secax = ax.secondary_xaxis('top')
    secax.set_xlabel("Sensing airtime fraction", color='gray')
    secax.set_xticks(main_ticks)
    secax.set_xticklabels(sensing_labels)
    secax.tick_params(axis='x', colors='gray')

handles, labels = ax.get_legend_handles_labels()
desired_order = ["baseline", "time", "tx", "rx", "tx+rx"]
sorted_pairs = sorted(zip(labels, handles), key=lambda x: desired_order.index(x[0]) if x[0] in desired_order else 999)
labels, handles = zip(*sorted_pairs)
ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 0.06), title="Subsampling approach")

fig.tight_layout()

# === Save or show figure ===
if args.savefig:
    plt.savefig(args.logdir.replace("/", "") + ("_actual" if USE_ACTUAL_S_VALS else "") + ".pdf",
                bbox_inches='tight', pad_inches=0)
else:
    plt.show()


# if args.logdir2:
#     # Helper function to collect all data ignoring epoch, with optional std
#     def collect_all_epochs(results, network, approach, subsample, return_std=False):
#         xs = []
#         ms = []
#         for (ep, net, subtime, subtx, subrx, app), acc_list in results.items():
#             if net != network or app != approach:
#                 continue
#             if subsample == "subtime" and app == "single" and subtx == 1 and subrx == 1:
#                 x = subtime
#             elif subsample == "subtx" and subtime == 1 and subrx == 1:
#                 x = subtx
#             elif subsample == "subrx" and subtime == 1 and subtx == 1:
#                 x = subrx
#             elif subsample == "all_beams" and subtime == 1 and subtx == subrx:
#                 x = subtx
#             else:
#                 continue
#             for acc in acc_list:
#                 xs.append(x)
#                 ms.append(acc)
#         if not xs:
#             if return_std:
#                 return np.array([]), np.array([]), np.array([])
#             else:
#                 return np.array([]), np.array([])
#         xs = np.array(xs)
#         ms = np.array(ms)
#         if USE_ACTUAL_S_VALS:
#             xs = adjust_x(subsample, xs)
#         elif subsample == "all_beams":
#             xs = xs ** 2
#         grouped = defaultdict(list)
#         for xval, val in zip(xs, ms):
#             grouped[round(xval, 6)].append(val)
#         xvals_sorted = sorted(grouped)
#         mvals = np.array([np.mean(grouped[xv]) for xv in xvals_sorted])
#         if return_std:
#             stdvals = np.array([np.std(grouped[xv]) for xv in xvals_sorted])
#             return np.array(xvals_sorted), mvals, stdvals
#         else:
#             return np.array(xvals_sorted), mvals
#
#     results1 = load_results(args.logdir)
#     results2 = load_results(args.logdir2)
#
#     # For simplicity, pick a network from results1 and results2
#     networks1 = sorted(set(k[1] for k in results1))
#     networks2 = sorted(set(k[1] for k in results2))
#     if not networks1 or not networks2:
#         print("No networks found in one of the logdirs.")
#     else:
#         network1 = networks1[0]
#         network2 = networks2[0]
#
#         fig_diff, ax_diff = plt.subplots(figsize=(12, 4))
#
#         added_labels = set()
#         offset_spacing = 0.1
#         tol = 1e-5  # Tolerance for matching x-values between logdirs
#
#         for subsample_idx, subsample in reversed(list(enumerate(subsample_types))):
#             approaches = ["single"] if subsample == "subtime" else ["repeated", "interleaved"]
#             for approach in approaches:
#                 if approach == "interleaved" and not SHOW_INTERLEAVED:
#                     continue
#
#                 x1, m1, std1 = collect_all_epochs(results1, network1, approach, subsample, return_std=True)
#                 x2, m2 = collect_all_epochs(results2, network2, approach, subsample)
#
#                 if len(x1) == 0 or len(x2) == 0:
#                     continue
#
#                 x_matched = []
#                 diffs = []
#                 stds_to_plot = []
#
#                 for xi, val1 in enumerate(x1):
#                     # find closest match in x2
#                     matches = [(j, val2) for j, val2 in enumerate(x2) if abs(val1 - val2) < tol]
#                     if matches:
#                         j = matches[0][0]
#                         diff = m1[xi] - m2[j]
#                         x_matched.append(val1)
#                         diffs.append(diff)
#                         stds_to_plot.append(std1[xi])
#
#                 if not diffs:
#                     continue
#
#                 x_matched = np.array(x_matched)
#                 diffs = np.array(diffs)
#                 stds_to_plot = np.array(stds_to_plot)
#
#                 offset = (subsample_idx - 1.5) * (offset_spacing / 2)
#                 xplot = np.where(x_matched != 1, x_matched + offset, x_matched)
#
#                 color = subsample_colors[subsample]
#                 label = subsample_labels[subsample] if (subsample_labels[subsample] not in added_labels) else None
#
#                 # Bars for difference
#                 ax_diff.bar(xplot, diffs, width=0.05, color=color, alpha=0.8, label=label)
#
#                 # Scatter for std from logdir1
#                 ax_diff.scatter(xplot, stds_to_plot, color=color, marker='o', s=50, edgecolor='black', zorder=5)
#
#                 if label:
#                     added_labels.add(label)
#
#         ax_diff.set_xlim(0.5, 9.5)
#         ax_diff.set_xlabel("Subsampling factor (actual)" if USE_ACTUAL_S_VALS else "Subsampling factor (target)")
#         ax_diff.set_ylabel("Difference in Accuracy (logdir1 - logdir2)")
#         ax_diff.grid(True)
#         ax_diff.axhline(0, color='black', linewidth=0.8, linestyle='--')
#
#         handles, labels = ax_diff.get_legend_handles_labels()
#         desired_order = ["time", "tx", "rx", "tx+rx"]
#         sorted_pairs = sorted(zip(labels, handles), key=lambda x: desired_order.index(x[0]) if x[0] in desired_order else 999)
#         if sorted_pairs:
#             labels, handles = zip(*sorted_pairs)
#             ax_diff.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 0.06), title="Subsampling approach")
#
#         fig_diff.tight_layout()
#
#         # Show second figure
#         plt.show()
