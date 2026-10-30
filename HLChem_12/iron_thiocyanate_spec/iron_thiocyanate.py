"""Iron Thiocyanate Lab: Tidy data, per-group overlays, tiered comparisons, and color-graded master plot."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Setup directories
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data") if os.path.exists(os.path.join(HERE, "data")) else HERE
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

# Define all group files mapped by concentration tier
HIGH_CONC_GROUPS = {
    "Gwyn, Ana & Amy": "Gwyn_Ana_Amy.csv",
    "Kevin J & Khoa": "KevinJ_Khoa.csv",
    "Zihan & Darren": "Zihan_Darren.csv",
    "Wilson & Ryan (Part 2)": "Wilson_Ryan_part2.csv"
}

LOW_CONC_GROUPS = {
    "Kevin L, Leo & Angus": "KevinL_Leo_Angus.csv",
    "Wilson & Ryan (Part 1)": "Wilson_Ryan_part1.csv"
}

ALL_GROUPS = {**HIGH_CONC_GROUPS, **LOW_CONC_GROUPS}

def parse_spectroscopy_csv(filepath):
    """Parses multi-run absorbance CSV files into a tidy DataFrame."""
    df = pd.read_csv(filepath)
    runs = []
    cols = df.columns.tolist()
    i = 0
    run_num = 1
    while i < len(cols) - 1:
        wav_col = cols[i]
        abs_col = cols[i+1]
        
        temp_df = df[[wav_col, abs_col]].dropna()
        temp_df.columns = ["wavelength_nm", "absorbance"]
        temp_df["run"] = f"Run {run_num}"
        runs.append(temp_df)
        
        i += 2
        run_num += 1
    return pd.concat(runs, ignore_index=True)

# Load and consolidate all group data
all_group_data = {}
master_rows = []

for group_name, filename in ALL_GROUPS.items():
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        gdf = parse_spectroscopy_csv(path)
        gdf["group"] = group_name
        all_group_data[group_name] = gdf
        master_rows.append(gdf)
    else:
        print(f"Warning: Could not find {path}")

if master_rows:
    master_df = pd.concat(master_rows, ignore_index=True)
    master_df.to_csv(os.path.join(HERE, "All_Groups_Spectroscopy_Data.csv"), index=False)

def clean_filename(name):
    return name.lower().replace(" & ", "_").replace(", ", "_").replace("(", "").replace(")", "").replace(" ", "_")

# 1. Per-group graphs: Overlay ALL runs on the same graph for each group
for group_name, gdf in all_group_data.items():
    fig, ax = plt.subplots(figsize=(9, 6))
    for run, r_df in gdf.groupby("run", sort=False):
        ax.plot(r_df["wavelength_nm"], r_df["absorbance"], label=run, linewidth=1.8)
    
    ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Absorbance", fontsize=11, fontweight="bold")
    ax.set_title(f"Iron Thiocyanate Absorbance Spectra — {group_name}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Runs / Solutions", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"group_{clean_filename(group_name)}_all_runs.png"), dpi=200)
    plt.close(fig)

# 2. Tiered Pooled Comparison Plots
tiers = [
    ("High-Concentration Tier", HIGH_CONC_GROUPS, "pooled_high_concentration.png"),
    ("Low-Concentration Tier", LOW_CONC_GROUPS, "pooled_low_concentration.png")
]

for tier_title, tier_dict, filename in tiers:
    fig, ax = plt.subplots(figsize=(10, 6))
    for group_name in tier_dict:
        if group_name in all_group_data:
            gdf = all_group_data[group_name]
            max_run = sorted(gdf["run"].unique(), key=lambda x: int(x.split(" ")[1]))[-1]
            r_df = gdf[gdf["run"] == max_run]
            ax.plot(r_df["wavelength_nm"], r_df["absorbance"], label=f"{group_name} ({max_run})", linewidth=1.8)

    ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Absorbance", fontsize=11, fontweight="bold")
    ax.set_title(f"Pooled Comparison — {tier_title}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Groups", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, filename), dpi=200)
    plt.close(fig)

# 3. Master Plot with Unique Group Colors & Run Gradients
fig, ax = plt.subplots(figsize=(12, 8))

group_base_colors = {
    "Gwyn, Ana & Amy": "#1f77b4",          # Blue
    "Kevin J & Khoa": "#2ca02c",           # Green
    "Kevin L, Leo & Angus": "#9467bd",     # Purple
    "Zihan & Darren": "#ff7f0e",           # Orange
    "Wilson & Ryan (Part 1)": "#8c564b",   # Brown
    "Wilson & Ryan (Part 2)": "#e377c2"    # Magenta
}

for group_name, gdf in all_group_data.items():
    base_color = group_base_colors.get(group_name, "#333333")
    runs = sorted(gdf["run"].unique(), key=lambda x: int(x.split(" ")[1]))
    num_runs = len(runs)
    
    for idx, run in enumerate(runs):
        r_df = gdf[gdf["run"] == run]
        fraction = (idx + 1) / max(num_runs, 1)
        alpha = 0.3 + 0.7 * fraction  # Gradient from transparent (run 1) to solid (final run)
        
        ax.plot(r_df["wavelength_nm"], r_df["absorbance"], 
                color=base_color, alpha=alpha, linewidth=1.5, 
                label=f"{group_name} ({run})")

ax.set_xlabel("Wavelength (nm)", fontsize=12, fontweight="bold")
ax.set_ylabel("Absorbance", fontsize=12, fontweight="bold")
ax.set_title("Master Plot: Distinct Group Colors & Run Gradients", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(title="Groups & Runs", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "master_gradient_all_runs.png"), dpi=250)
plt.close(fig)

print("All graphs successfully exported to 'plots/' folder.")