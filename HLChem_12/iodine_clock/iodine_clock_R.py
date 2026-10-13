"""Lab 6.1B - Iodine Clock: tidy data + per-group and pooled graphs
with a linear line of best fit, equation, R and R^2 on every graph."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))  # folder this script lives in
OUT = HERE                                        # CSVs saved next to the script
PLOTS = os.path.join(HERE, "plots_R")               # graphs saved in <script folder>/plots
os.makedirs(PLOTS, exist_ok=True)

# group -> ([(conc_M, time_s)], [(temp_C, time_s)])
RAW = {
    "Angus & Darren": (
        [(0.020, 27.77), (0.018, 30.07), (0.016, 31.34), (0.012, 39.16), (0.008, 51.44)],
        [(10.0, 15.92), (15.0, 30.66), (22.5, 27.77), (38.0, 67.61)]),
    "Sophia & Ryan": (
        [(0.020, 26.38), (0.016, 33.20), (0.012, 41.00), (0.008, 63.37), (0.004, 136.55)],
        [(9.0, 32.5), (23.0, 39.5)]),
    "Lucas & Ana": (
        [(0.020, 23.0), (0.018, 28.0), (0.016, 45.0), (0.014, 53.0), (0.012, 61.0)],
        [(13.0, 12.0), (20.0, 34.0), (24.5, 23.0), (36.5, 598.0)]),
    "Kevin J & Amy": (
        [(0.020, 18.0), (0.016, 20.0), (0.012, 26.0), (0.008, 40.0), (0.004, 80.0)],
        [(11.0, 41.42), (15.0, 35.58), (21.0, 23.67)]),
    "Leo & Kevin L": (
        [(0.020, 15.0), (0.016, 25.0), (0.012, 25.0), (0.010, 37.0), (0.008, 53.0), (0.004, 94.47)],
        [(24.0, 37.0), (10.0, 60.0)]),
    "Howard & Grace": (
        [(0.020, 28.0), (0.016, 35.11), (0.012, 37.20), (0.008, 50.23), (0.004, 102.21)],
        [(7.0, 24.21), (22.0, 28.0), (35.0, 29.5), (45.0, 33.45)]),
    "Andison & Khoa": (
        [(0.020, 20.0), (0.016, 24.0), (0.012, 30.0), (0.008, 45.0), (0.004, 90.0)],
        [(22.0, 20.0), (8.0, 21.0), (30.0, 27.0)]),
}

rows = []
for group, (conc, temp) in RAW.items():
    for x, t in conc:
        rows.append(dict(group=group, part="I_concentration", x=x, time_s=t))
    for x, t in temp:
        rows.append(dict(group=group, part="II_temperature", x=x, time_s=t))
df = pd.DataFrame(rows)
df["rate_per_s"] = 1.0 / df["time_s"]
df = df.sort_values(["part", "group", "x"]).reset_index(drop=True)

p1 = df[df.part == "I_concentration"].rename(columns={"x": "KIO3_conc_M"}).drop(columns="part")
p2 = df[df.part == "II_temperature"].rename(columns={"x": "temperature_C"}).drop(columns="part")
df.to_csv(f"{OUT}/Iodine_Clock_Lab6.1B_All_Data.csv", index=False)
p1.to_csv(f"{OUT}/Iodine_Clock_Lab6.1B_Part1_Concentration.csv", index=False)
p2.to_csv(f"{OUT}/Iodine_Clock_Lab6.1B_Part2_Temperature.csv", index=False)

# part key, x-axis label, short title, marker
PARTS = [
    ("I_concentration", "[KIO$_3$] (M)", "Rate vs [KIO$_3$]", "o"),
    ("II_temperature", "Temperature (°C)", "Rate vs Temperature", "s"),
]


def fname(s):
    return s.replace(" & ", "_and_").replace(" ", "_")


def best_fit(x, y):
    """Least-squares line y = m*x + b. Returns m, b, r, r2 (None if < 2 points)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    m, b = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return m, b, r, r * r


def add_fit(ax, x, y, color="black"):
    """Draw the best-fit line and return a legend handle labelled with equation, R, R^2."""
    fit = best_fit(x, y)
    if fit is None:
        return None
    m, b, r, r2 = fit
    xs = np.array([np.min(x), np.max(x)])
    note = " (n=2, exact fit)" if len(x) == 2 else ""
    label = f"y = {m:.3g}x {'+' if b >= 0 else '-'} {abs(b):.3g}\nR = {r:.3f}, R² = {r2:.3f}{note}"
    (line,) = ax.plot(xs, m * xs + b, linestyle="--", color=color, linewidth=1.6, label=label)
    return line


# Per-group graphs (two panels)
for group in RAW:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (part, xl, ttl, mk) in zip(axes, PARTS):
        d = df[(df.group == group) & (df.part == part)]
        ax.scatter(d.x, d.rate_per_s, marker=mk, s=45, zorder=3)
        add_fit(ax, d.x, d.rate_per_s, color="tab:red")
        ax.set_xlabel(xl)
        ax.set_ylabel("Rate (s$^{-1}$)")
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(group, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"{fname(group)}.png"), dpi=200)
    plt.close(fig)

# Pooled graphs: points coloured by group, one best-fit line through all points
for part, xl, ttl, mk in PARTS:
    fig, ax = plt.subplots(figsize=(10, 6))
    group_handles = []
    for group in RAW:
        d = df[(df.group == group) & (df.part == part)]
        h = ax.scatter(d.x, d.rate_per_s, marker=mk, s=45, label=group, zorder=3)
        group_handles.append(h)
    allp = df[df.part == part]
    fit_line = add_fit(ax, allp.x, allp.rate_per_s, color="black")
    ax.set_xlabel(xl)
    ax.set_ylabel("Rate (s$^{-1}$)")
    ax.set_title(f"Pooled: {ttl}", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fit_legend = ax.legend(handles=[fit_line], loc="best", fontsize=9, title="All groups")
    ax.add_artist(fit_legend)
    ax.legend(handles=group_handles, title="Group members",
              bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"pooled_{part}.png"), dpi=200)
    plt.close(fig)

print(df.groupby(["part", "group"]).size())