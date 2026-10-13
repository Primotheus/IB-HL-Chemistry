"""Lab 6.1B - Iodine Clock: tidy data + per-group and pooled graphs."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))  # folder this script lives in
OUT = HERE                                        # CSVs saved next to the script
PLOTS = os.path.join(HERE, "plots")               # graphs saved in <script folder>/plots
os.makedirs(PLOTS, exist_ok=True)

# group -> (members, [(conc_M, time_s)], [(temp_C, time_s)])
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

PARTS = [
    ("I_concentration", "[KIO$_3$] initial concentration (M)", "Effect of [KIO$_3$] on Rate", "o"),
    ("II_temperature", "Temperature (°C)", "Effect of Temperature on Rate", "s"),
]

def fname(s):
    return s.replace(" & ", "_and_").replace(" ", "_")

# Per-group graph (two panels), titled with member names
for group in RAW:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (part, xl, ttl, m) in zip(axes, PARTS):
        d = df[(df.group == group) & (df.part == part)]
        ax.plot(d.x, d.rate_per_s, marker=m, linestyle="-")
        ax.set_xlabel(xl)
        ax.set_ylabel("Rate (1/time, s$^{-1}$)")
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Lab 6.1B Iodine Clock - {group}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"{fname(group)}.png"), dpi=200)
    plt.close(fig)

# Pooled graphs (sorted by x so lines don't zigzag)
for part, xl, ttl, m in PARTS:
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in RAW:
        d = df[(df.group == group) & (df.part == part)]
        ax.plot(d.x, d.rate_per_s, marker=m, linestyle="-", label=group)
    ax.set_xlabel(xl)
    ax.set_ylabel("Rate (1/time, s$^{-1}$)")
    ax.set_title(f"Lab 6.1B - {ttl} (Pooled Data)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Group members", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"pooled_{part}.png"), dpi=200)
    plt.close(fig)

print(df.groupby(["part", "group"]).size())