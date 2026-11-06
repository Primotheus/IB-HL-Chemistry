
import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# SETTINGS
# ==========================================

MEASUREMENT_WAVELENGTH = 628  # nm
PEAK_MIN = 400
PEAK_MAX = 700

# Standards: 10%, 20%, 30%, 40%, 50%, 70%, 100% of Stock B
# Stock B = 10% of Stock A
# Therefore, these equal 1%, 2%, 3%, 4%, 5%, 7%, 10% of Stock A.

CONCENTRATIONS = np.array([1, 2, 3, 4, 5, 7, 10])

STANDARD_NAMES = [
    "Standard 1", "Standard 2", "Standard 3",
    "Standard 4", "Standard 5", "Standard 6",
    "Standard 7"
]


# ==========================================
# SELECT CSV FILE
# ==========================================

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select spectrophotometer CSV file",
    filetypes=[("CSV files", "*.csv")]
)

if not file_path:
    raise SystemExit("No file selected.")

data = pd.read_csv(file_path)

output_folder = os.path.splitext(file_path)[0] + "_Analysis"
os.makedirs(output_folder, exist_ok=True)


# ==========================================
# EXTRACT DATA
# ==========================================

spectra = []

for i in range(1, 8):

    wavelength_col = f"Data Set {i}:Wavelength(nm)"
    absorbance_col = f"Data Set {i}:Absorbance"

    if wavelength_col not in data.columns or absorbance_col not in data.columns:
        raise ValueError(
            f"Missing expected columns for Data Set {i}. "
            "Check the CSV column names."
        )

    spectrum = pd.DataFrame({
        "Wavelength": pd.to_numeric(
            data[wavelength_col], errors="coerce"
        ),
        "Absorbance": pd.to_numeric(
            data[absorbance_col], errors="coerce"
        )
    }).dropna()

    spectrum = spectrum.sort_values("Wavelength")
    spectra.append(spectrum)


# ==========================================
# CALCULATE PEAKS AND ABSORBANCE AT 628 NM
# ==========================================

results = []

for i, spectrum in enumerate(spectra):

    visible = spectrum[
        (spectrum["Wavelength"] >= PEAK_MIN) &
        (spectrum["Wavelength"] <= PEAK_MAX)
    ]

    peak_index = visible["Absorbance"].idxmax()

    peak_wavelength = visible.loc[peak_index, "Wavelength"]
    peak_absorbance = visible.loc[peak_index, "Absorbance"]

    # Use the nearest recorded wavelength to 628 nm
    nearest_index = (
        spectrum["Wavelength"] - MEASUREMENT_WAVELENGTH
    ).abs().idxmin()

    actual_wavelength = spectrum.loc[
        nearest_index, "Wavelength"
    ]

    absorbance_628 = spectrum.loc[
        nearest_index, "Absorbance"
    ]

    results.append({
        "Standard": STANDARD_NAMES[i],
        "Concentration (% Stock A)": CONCENTRATIONS[i],
        "Peak wavelength (nm)": peak_wavelength,
        "Maximum absorbance": peak_absorbance,
        "Measurement wavelength used (nm)": actual_wavelength,
        "Absorbance at measurement wavelength": absorbance_628
    })


results_df = pd.DataFrame(results)

x = results_df["Concentration (% Stock A)"].to_numpy()
y = results_df["Absorbance at measurement wavelength"].to_numpy()


# ==========================================
# LINEAR REGRESSION AND R-SQUARED
# ==========================================

slope, intercept = np.polyfit(x, y, 1)

predicted = slope * x + intercept
residuals = y - predicted

ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)

r_squared = 1 - (ss_res / ss_tot)

results_df["Predicted absorbance"] = predicted
results_df["Residual"] = residuals

equation = (
    f"A = {slope:.6f}C "
    f"{intercept:+.6f}"
)


# ==========================================
# GRAPH 1: ALL ABSORPTION SPECTRA
# ==========================================

plt.figure(figsize=(11, 7))

for i, spectrum in enumerate(spectra):
    plt.plot(
        spectrum["Wavelength"],
        spectrum["Absorbance"],
        label=f"{STANDARD_NAMES[i]} ({CONCENTRATIONS[i]}%)"
    )

plt.xlim(380, 750)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title("Absorption Spectra of All Seven Standards")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "01_All_Absorption_Spectra.png"),
    dpi=300
)
plt.close()


# ==========================================
# GRAPH 2: INDIVIDUAL ABSORPTION SPECTRA
# ==========================================

for i, spectrum in enumerate(spectra):

    plt.figure(figsize=(9, 6))

    plt.plot(
        spectrum["Wavelength"],
        spectrum["Absorbance"],
        label=STANDARD_NAMES[i]
    )

    peak = results_df.iloc[i]

    plt.axvline(
        peak["Peak wavelength (nm)"],
        linestyle="--",
        label=f"Peak: {peak['Peak wavelength (nm)']:.1f} nm"
    )

    plt.axvline(
        MEASUREMENT_WAVELENGTH,
        linestyle=":",
        label="Selected wavelength: 628 nm"
    )

    plt.xlim(380, 750)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.title(f"Absorption Spectrum - {STANDARD_NAMES[i]}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            f"02_Spectrum_Standard_{i+1}.png"
        ),
        dpi=300
    )

    plt.close()


# ==========================================
# GRAPH 3: CALIBRATION CURVE
# ==========================================

x_line = np.linspace(min(x), max(x), 200)
y_line = slope * x_line + intercept

plt.figure(figsize=(9, 6))

plt.scatter(
    x, y,
    s=65,
    label="Experimental data"
)

plt.plot(
    x_line, y_line,
    linestyle="--",
    label="Linear regression"
)

plt.xlabel("Relative concentration (% of Stock A)")
plt.ylabel("Absorbance at 628 nm")
plt.title("Calibration Curve at 628 nm")

plt.text(
    0.05, 0.95,
    f"{equation}\n$R^2$ = {r_squared:.6f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "03_Calibration_Curve.png"),
    dpi=300
)
plt.close()


# ==========================================
# GRAPH 4: RESIDUAL PLOT
# ==========================================

plt.figure(figsize=(9, 6))

plt.scatter(x, residuals, s=65)

plt.axhline(
    0,
    linestyle="--",
    label="Zero residual"
)

plt.xlabel("Relative concentration (% of Stock A)")
plt.ylabel("Residual (observed - predicted absorbance)")
plt.title("Calibration Curve Residual Plot")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "04_Residual_Plot.png"),
    dpi=300
)
plt.close()


# ==========================================
# GRAPH 5: PEAK WAVELENGTH COMPARISON
# ==========================================

plt.figure(figsize=(9, 6))

plt.plot(
    x,
    results_df["Peak wavelength (nm)"],
    marker="o"
)

plt.axhline(
    MEASUREMENT_WAVELENGTH,
    linestyle="--",
    label="Selected measurement wavelength"
)

plt.xlabel("Relative concentration (% of Stock A)")
plt.ylabel("Peak wavelength (nm)")
plt.title("Peak Wavelength vs Relative Concentration")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "05_Peak_Wavelength.png"),
    dpi=300
)
plt.close()


# ==========================================
# EXPORT RESULTS
# ==========================================

results_df.to_csv(
    os.path.join(output_folder, "Analysis_Results.csv"),
    index=False
)


# ==========================================
# EXPORT REGRESSION SUMMARY
# ==========================================

with open(
    os.path.join(output_folder, "Regression_Summary.txt"),
    "w"
) as file:

    file.write("SPECTROPHOTOMETER CALIBRATION ANALYSIS\n")
    file.write("=" * 45 + "\n\n")

    file.write(f"Measurement wavelength: {MEASUREMENT_WAVELENGTH} nm\n")
    file.write("Concentration basis: Relative % of Stock A\n\n")

    file.write("LINEAR REGRESSION\n")
    file.write("-" * 25 + "\n")
    file.write(f"Equation: {equation}\n")
    file.write(f"Slope: {slope:.8f}\n")
    file.write(f"Intercept: {intercept:.8f}\n")
    file.write(f"R-squared: {r_squared:.8f}\n\n")

    file.write("INTERPRETATION\n")
    file.write("-" * 25 + "\n")
    file.write(
        "R-squared describes how closely the experimental "
        "data fit the linear regression model.\n"
    )
    file.write(
        "It does not independently prove that the calibration "
        "is accurate or that the relationship is linear "
        "throughout other concentration ranges.\n\n"
    )

    file.write("DATA POINTS\n")
    file.write("-" * 25 + "\n")

    for i in range(len(x)):
        file.write(
            f"{STANDARD_NAMES[i]}: "
            f"Concentration = {x[i]}%, "
            f"Absorbance = {y[i]:.6f}, "
            f"Predicted = {predicted[i]:.6f}, "
            f"Residual = {residuals[i]:.6f}\n"
        )


# ==========================================
# FINISHED
# ==========================================

print("\nANALYSIS COMPLETE")
print("=" * 40)
print(f"Calibration equation: {equation}")
print(f"R-squared: {r_squared:.6f}")
print(f"\nFiles exported to:\n{output_folder}")
print("\nGenerated:")
print("1. All absorption spectra")
print("2. Seven individual spectra")
print("3. Calibration curve")
print("4. Residual plot")
print("5. Peak wavelength comparison")
print("6. Analysis results CSV")
print("7. Regression summary TXT")
