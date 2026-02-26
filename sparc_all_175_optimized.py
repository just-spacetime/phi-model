#!/usr/bin/env python3
"""
SPARC All-175 Galaxies Dashboard
=================================
Generates a massive plot with 175 rows × 4 columns:
  Column 1: Rotation curve (Vobs, Vbar, MOND, PHI-Model)
  Column 2: Boost factor (Vobs² / Vbar²)
  Column 3: 1/|phi| vs radius
  Column 4: (1/phi)' vs radius

Usage:
  python sparc_all_175.py

Requires:
  - numpy, matplotlib
  - sparc_data/ folder with *_rotmod.dat files in same directory

Output:
  - sparc_all_175.png (saved to disk, no screen display)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob, sys

# --- CONFIGURATION ---
SPARC_DIR = './sparc_data'
Y_STAR = 0.5
A0_MOND = 1.2e-10       # MOND acceleration scale [m/s²]
C_PHI = 0.8e-5           # Our universal constant
OUTPUT_FILE = 'sparc_all_175.png'
DPI = 100
ROW_HEIGHT = 2.8
COL_WIDTH = 6.5
KPC_TO_M = 3.086e19
G_CONST = 6.67430e-11

# --- CHECK DATA ---
if not os.path.isdir(SPARC_DIR):
    print(f"ERROR: Folder '{SPARC_DIR}' not found!")
    print(f"  Put the SPARC _rotmod.dat files in a folder called 'sparc_data'")
    print(f"  next to this script, or edit SPARC_DIR at the top.")
    sys.exit(1)

files = sorted(glob.glob(os.path.join(SPARC_DIR, '*_rotmod.dat')))
print(f"Found {len(files)} SPARC files in {SPARC_DIR}")

if len(files) == 0:
    print("ERROR: No *_rotmod.dat files found!")
    sys.exit(1)

# --- READ FUNCTION ---
def read_sparc(filepath):
    rad, vobs, errv, vgas, vdisk, vbul = [], [], [], [], [], []
    distance = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('# Distance'):
                try:
                    distance = float(line.split('=')[1].replace('Mpc','').strip())
                except: pass
                continue
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    rad.append(float(parts[0]))
                    vobs.append(float(parts[1]))
                    errv.append(float(parts[2]))
                    vgas.append(float(parts[3]))
                    vdisk.append(float(parts[4]))
                    vbul.append(float(parts[5]))
                except ValueError:
                    continue
    return {
        'name': os.path.basename(filepath).replace('_rotmod.dat',''),
        'distance': distance,
        'Rad': np.array(rad),
        'Vobs': np.array(vobs),
        'errV': np.array(errv),
        'Vgas': np.array(vgas),
        'Vdisk': np.array(vdisk),
        'Vbul': np.array(vbul),
    }

def compute_potential(r_kpc, v2_bar_kms2):
    """Compute |phi(r)| and a_n(r) from baryonic v² profile."""
    r_m = r_kpc * KPC_TO_M
    v2_si = np.abs(v2_bar_kms2) * 1e6
    an = v2_si / r_m

    phi = np.zeros_like(r_m)
    for j in range(len(r_m)-2, -1, -1):
        dr = r_m[j+1] - r_m[j]
        phi[j] = phi[j+1] - 0.5*(an[j] + an[j+1]) * dr

    GM_est = an[-1] * r_m[-1]**2
    phi_boundary = -GM_est / r_m[-1]
    phi = phi + phi_boundary - phi[-1]

    return np.abs(phi), an

def mond_velocity(r_kpc, an):
    """Simple MOND: a*mu(a/a0)=a_n, mu(x)=x/(1+x)."""
    a_mond = 0.5 * (an + np.sqrt(an**2 + 4 * an * A0_MOND))
    r_m = r_kpc * KPC_TO_M
    return np.sqrt(a_mond * r_m) / 1000

def phi_model_velocity(r_kpc, an):
    """New phi-model: a_tot = a_n * (1 + C / sqrt(a_n))"""
    boost = 1.0 + C_PHI / np.sqrt(an)
    boost = np.maximum(boost, 0.1)  # safety
    r_m = r_kpc * KPC_TO_M
    return np.sqrt(an * boost * r_m) / 1000

def phi_model_velocity_with_c(r_kpc, an, c_value):
    """Phi-model with custom C parameter for optimization."""
    boost = 1.0 + c_value / np.sqrt(an)
    boost = np.maximum(boost, 0.1)  # safety
    r_m = r_kpc * KPC_TO_M
    return np.sqrt(an * boost * r_m) / 1000

def optimize_c_parameter(all_gals):
    """
    Find optimal C parameter that best fits phi-model to observed velocities.
    Uses chi-squared minimization across all galaxies.
    """
    print("\n" + "="*60)
    print("OPTIMIZING C PARAMETER FOR PHI-MODEL")
    print("="*60)
    
    def compute_chi2(c_value):
        """Compute total chi-squared for given C value across all galaxies."""
        total_chi2 = 0
        n_points = 0
        
        for d in all_gals:
            rad = d['Rad']
            vobs = d['Vobs']
            errv = d['errV']
            vgas = d['Vgas']
            vdisk = d['Vdisk']
            vbul = d['Vbul']
            
            # Baryonic velocity
            v2_gas = vgas * np.abs(vgas)
            v2_disk = Y_STAR * vdisk * np.abs(vdisk)
            v2_bul = Y_STAR * vbul * np.abs(vbul)
            v2_bar = v2_gas + v2_disk + v2_bul
            
            # Compute potential and acceleration
            mask_pot = (v2_bar > 0) & (rad > 0.01)
            if np.sum(mask_pot) >= 3:
                phi_calc, an_calc = compute_potential(rad[mask_pot], v2_bar[mask_pot])
                
                # Compute phi-model velocity with test C value
                v_phi_test = phi_model_velocity_with_c(rad[mask_pot], an_calc, c_value)
                
                # Only use points where we have valid predictions and observations
                mask_valid = (v_phi_test > 0) & np.isfinite(v_phi_test) & (vobs[mask_pot] > 0) & (errv[mask_pot] > 0)
                
                if np.sum(mask_valid) > 0:
                    vobs_valid = vobs[mask_pot][mask_valid]
                    errv_valid = errv[mask_pot][mask_valid]
                    v_phi_valid = v_phi_test[mask_valid]
                    
                    # Chi-squared calculation
                    chi2 = np.sum(((vobs_valid - v_phi_valid) / errv_valid)**2)
                    total_chi2 += chi2
                    n_points += np.sum(mask_valid)
        
        return total_chi2, n_points
    
    # Test range of C values
    c_values = np.arange(0.7e-5, 1.3e-5 + 0.01e-5, 0.01e-5)  # From 0.7e-5 to 1.3e-5 in steps of 0.01e-5
    chi2_values = []
    
    print(f"Testing {len(c_values)} different C values...")
    for i, c_test in enumerate(c_values):
        chi2, n_pts = compute_chi2(c_test)
        chi2_values.append(chi2)
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(c_values)}, C={c_test:.2e}, χ²={chi2:.2e}")
    
    chi2_values = np.array(chi2_values)
    
    # Find minimum
    min_idx = np.argmin(chi2_values)
    optimal_c = c_values[min_idx]
    min_chi2, n_total = compute_chi2(optimal_c)
    reduced_chi2 = min_chi2 / n_total if n_total > 0 else np.inf
    
    print("\n" + "-"*60)
    print("OPTIMIZATION RESULTS:")
    print(f"  Optimal C = {optimal_c:.6e}")
    print(f"  Minimum χ² = {min_chi2:.2e}")
    print(f"  Total data points = {n_total}")
    print(f"  Reduced χ² = {reduced_chi2:.3f}")
    print(f"  Original C was = {C_PHI:.6e}")
    print("-"*60)
    
    # Also compute chi2 for original C
    orig_chi2, _ = compute_chi2(C_PHI)
    print(f"  χ² with original C = {orig_chi2:.2e}")
    print(f"  Improvement factor = {orig_chi2/min_chi2:.2f}x")
    print("="*60 + "\n")
    
    return optimal_c, c_values, chi2_values

# --- LOAD ALL ---
print("Loading all galaxies...")
all_gals = []
for filepath in files:
    d = read_sparc(filepath)
    if len(d['Rad']) >= 3:
        all_gals.append(d)

n_gal = len(all_gals)
print(f"Loaded {n_gal} galaxies with >= 3 data points")

# --- OPTIMIZE C PARAMETER ---
optimal_c, c_test_values, chi2_results = optimize_c_parameter(all_gals)
print(f"Using optimized C = {optimal_c:.6e} for plots")

# --- CREATE FIGURE ---
print(f"Creating figure: {n_gal} rows x 4 columns ...")
fig, axes = plt.subplots(n_gal, 4, figsize=(4 * COL_WIDTH, n_gal * ROW_HEIGHT))

if n_gal == 1:
    axes = axes.reshape(1, 4)

for i, d in enumerate(all_gals):
    if (i + 1) % 25 == 0 or i == 0:
        print(f"  Processing {i+1}/{n_gal}: {d['name']}")

    rad = d['Rad']
    vobs = d['Vobs']
    errv = d['errV']
    vgas = d['Vgas']
    vdisk = d['Vdisk']
    vbul = d['Vbul']

    # Baryonic velocity
    v2_gas = vgas * np.abs(vgas)
    v2_disk = Y_STAR * vdisk * np.abs(vdisk)
    v2_bul = Y_STAR * vbul * np.abs(vbul)
    v2_bar = v2_gas + v2_disk + v2_bul
    v_bar = np.sqrt(np.maximum(v2_bar, 0))

    # Potential and acceleration
    mask_pot = (v2_bar > 0) & (rad > 0.01)
    phi = np.full_like(rad, np.nan)
    an = np.full_like(rad, np.nan)
    inv_phi = np.full_like(rad, np.nan)
    d_inv_phi = np.full_like(rad, np.nan)
    v_mond = np.full_like(rad, np.nan)
    v_phi_model = np.full_like(rad, np.nan)

    if np.sum(mask_pot) >= 3:
        phi_calc, an_calc = compute_potential(rad[mask_pot], v2_bar[mask_pot])
        phi[mask_pot] = phi_calc
        an[mask_pot] = an_calc

        inv_phi_calc = 1.0 / phi_calc
        inv_phi[mask_pot] = inv_phi_calc

        d_inv_phi_calc = np.gradient(inv_phi_calc, rad[mask_pot] * KPC_TO_M)
        d_inv_phi[mask_pot] = d_inv_phi_calc

        # MOND
        v_mond[mask_pot] = mond_velocity(rad[mask_pot], an_calc)

        # New phi model (simplified - only depends on a_n)
        v_phi_model[mask_pot] = phi_model_velocity_with_c(rad[mask_pot], an_calc, optimal_c)

    dist_str = f"D={d['distance']:.1f} Mpc" if d['distance'] else ""

    # ========== COL 1: Rotation Curve ==========
    ax1 = axes[i, 0]
    ax1.errorbar(rad, vobs, yerr=errv, fmt='o', color='black', ms=3, lw=0.8,
                 capsize=1.5, zorder=10, label='$V_{obs}$')
    ax1.plot(rad, np.abs(vdisk) * np.sqrt(Y_STAR), 'r-', lw=1, alpha=0.5,
             label=f'$\\sqrt{{Y_*}}V_{{disk}}$')
    ax1.plot(rad, np.abs(vgas), 'b--', lw=1, alpha=0.5, label='$V_{gas}$')
    if np.any(np.abs(vbul) > 1):
        ax1.plot(rad, np.abs(vbul) * np.sqrt(Y_STAR), 'm:', lw=0.8, alpha=0.4,
                 label='$\\sqrt{Y_*}V_{bul}$')
    ax1.plot(rad, v_bar, 'g-', lw=1.5, label='$V_{bar}$')

    # MOND
    mask_mond = np.isfinite(v_mond) & (v_mond > 0)
    if np.sum(mask_mond) > 2:
        ax1.plot(rad[mask_mond], v_mond[mask_mond], color='orange', ls='--',
                 lw=1.5, alpha=0.8, label='MOND')

    # PHI MODEL
    mask_pm = np.isfinite(v_phi_model) & (v_phi_model > 0)
    if np.sum(mask_pm) > 2:
        ax1.plot(rad[mask_pm], v_phi_model[mask_pm], color='blue', ls='-',
                 lw=2, alpha=0.9, label='$\\phi$-Model')

    ax1.set_ylabel('v [km/s]', fontsize=7)
    ax1.set_title(f'{d["name"]}  ({dist_str}, N={len(rad)})',
                  fontsize=8, fontweight='bold', loc='left')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.2)
    if i == 0:
        ax1.legend(fontsize=5, ncol=4, loc='upper right')
    if i == n_gal - 1:
        ax1.set_xlabel('r [kpc]', fontsize=8)

    # ========== COL 2: Boost Factor ==========
    ax2 = axes[i, 1]
    mask_b = v_bar > 5
    if np.sum(mask_b) > 2:
        boost = (vobs[mask_b] / v_bar[mask_b])**2
        ax2.scatter(rad[mask_b], boost, s=12, c='darkred', zorder=5)
        ax2.axhline(1, color='black', ls=':', alpha=0.5)
        ax2.fill_between([0, rad.max()*1.1], 0, 1, color='lightblue',
                         alpha=0.15, zorder=0)
        ax2.set_ylim(bottom=0, top=min(max(boost)*1.3, 50))
    else:
        ax2.text(0.5, 0.5, 'insufficient data', transform=ax2.transAxes,
                 ha='center', fontsize=7, color='gray')
    ax2.set_ylabel('Boost', fontsize=7)
    ax2.set_title('$V_{obs}^2 / V_{bar}^2$', fontsize=7, loc='left', color='darkred')
    ax2.grid(True, alpha=0.2)
    if i == n_gal - 1:
        ax2.set_xlabel('r [kpc]', fontsize=8)

    # ========== COL 3: 1/|phi| ==========
    ax3 = axes[i, 2]
    mask_phi = np.isfinite(inv_phi) & (inv_phi > 0)
    if np.sum(mask_phi) > 2:
        ax3.plot(rad[mask_phi], inv_phi[mask_phi], color='navy', lw=1.5)
    else:
        ax3.text(0.5, 0.5, 'insufficient data', transform=ax3.transAxes,
                 ha='center', fontsize=7, color='gray')
    ax3.set_ylabel('$1/|\\phi|$', fontsize=7)
    ax3.set_title('$1/|\\phi|$', fontsize=7, loc='left', color='navy')
    ax3.grid(True, alpha=0.2)
    ax3.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2))
    if i == n_gal - 1:
        ax3.set_xlabel('r [kpc]', fontsize=8)

    # ========== COL 4: (1/phi)' ==========
    ax4 = axes[i, 3]
    mask_dp = np.isfinite(d_inv_phi)
    if np.sum(mask_dp) > 2:
        ax4.plot(rad[mask_dp], d_inv_phi[mask_dp], color='green', lw=1.5)
        vals = d_inv_phi[mask_dp]
        n4 = max(1, len(vals)//4)
        fp_mean_plot = np.mean(np.abs(vals[n4:]))
        ax4.axhline(fp_mean_plot, color='red', ls=':', alpha=0.7, lw=1)
        ax4.text(0.95, 0.90, f'mean={fp_mean_plot:.2e}', transform=ax4.transAxes,
                 fontsize=5, ha='right', color='red')
    else:
        ax4.text(0.5, 0.5, 'insufficient data', transform=ax4.transAxes,
                 ha='center', fontsize=7, color='gray')
    ax4.set_ylabel("$(1/\\phi)'$", fontsize=7)
    ax4.set_title("$(1/\\phi)'$", fontsize=7, loc='left', color='green')
    ax4.grid(True, alpha=0.2)
    ax4.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2))
    if i == n_gal - 1:
        ax4.set_xlabel('r [kpc]', fontsize=8)

# --- FINALIZE ---
print("Adjusting layout...")
plt.suptitle(
    f'SPARC: {n_gal} Galaxies  |  '
    f'$V_{{bar}}^2 = V_{{gas}}^2 + {Y_STAR} V_{{disk}}^2 + {Y_STAR} V_{{bul}}^2$  |  '
    f'MOND ($a_0$={A0_MOND:.1e})  |  '
    f'$\\phi$-Model: $a_{{tot}} = a_n(1 + C/\\sqrt{{a_n}})$, C={optimal_c:.2e} (optimized)',
    fontsize=13, fontweight='bold', y=1.0 - 0.2/n_gal)

plt.subplots_adjust(hspace=0.45, wspace=0.30, top=1.0 - 0.5/n_gal,
                    bottom=0.005, left=0.04, right=0.98)

print(f"Saving to {OUTPUT_FILE} (DPI={DPI})...")
print(f"  This may take a while for {n_gal} galaxies...")
fig.savefig(OUTPUT_FILE, dpi=DPI)
plt.close(fig)

print(f"\nDone! Output saved as: {OUTPUT_FILE}")