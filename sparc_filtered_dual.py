import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob

G_const = 6.67430e-11
M_sun = 1.989e30
KPC_TO_M = 3.086e19
Y_STAR = 0.5

SPARC_DIR = './sparc_data'
if not os.path.isdir(SPARC_DIR):
    SPARC_DIR = '/home/claude/sparc_data'

files = sorted(glob.glob(os.path.join(SPARC_DIR, '*_rotmod.dat')))
print(f"Found {len(files)} files")

def read_sparc(filepath):
    rad, vobs, errv, vgas, vdisk, vbul = [], [], [], [], [], []
    distance = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('# Distance'):
                try: distance = float(line.split('=')[1].replace('Mpc','').strip())
                except: pass
                continue
            if line.startswith('#') or not line: continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    rad.append(float(parts[0])); vobs.append(float(parts[1]))
                    errv.append(float(parts[2])); vgas.append(float(parts[3]))
                    vdisk.append(float(parts[4])); vbul.append(float(parts[5]))
                except: continue
    return {
        'name': os.path.basename(filepath).replace('_rotmod.dat',''),
        'distance': distance,
        'Rad': np.array(rad), 'Vobs': np.array(vobs), 'errV': np.array(errv),
        'Vgas': np.array(vgas), 'Vdisk': np.array(vdisk), 'Vbul': np.array(vbul),
    }

def compute_potential(r_kpc, v2_bar_kms2):
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

def analyze_galaxy(d):
    """Returns result dict or None if galaxy fails basic criteria."""
    if len(d['Rad']) < 8: return None
    
    rad = d['Rad']; vobs = d['Vobs']
    v2_gas = d['Vgas'] * np.abs(d['Vgas'])
    v2_disk = Y_STAR * d['Vdisk'] * np.abs(d['Vdisk'])
    v2_bul = Y_STAR * d['Vbul'] * np.abs(d['Vbul'])
    v2_bar = v2_gas + v2_disk + v2_bul
    v_bar = np.sqrt(np.maximum(v2_bar, 0))
    
    mask = (v2_bar > 0) & (vobs > 10) & (rad > 0.1) & (v_bar > 5)
    if np.sum(mask) < 8: return None
    
    r_m = rad[mask]; vo = vobs[mask]; vb = v_bar[mask]
    phi, an = compute_potential(r_m, v2_bar[mask])
    inv_phi = 1.0 / phi
    boost = (vo / vb)**2
    
    # Total baryonic mass
    r_last_m = r_m[-1] * KPC_TO_M
    GM_bar = np.abs(v2_bar[mask][-1]) * 1e6 * r_last_m
    
    # Full linear fit (free intercept)
    coeffs_full = np.polyfit(inv_phi, boost, 1)
    intercept = coeffs_full[1]
    
    # Forced fit in 20-80% range
    r_lo = r_m.min() + 0.2 * (r_m.max() - r_m.min())
    r_hi = r_m.min() + 0.8 * (r_m.max() - r_m.min())
    mid = (r_m >= r_lo) & (r_m <= r_hi)
    if np.sum(mid) < 4: return None
    
    b_m1 = boost[mid] - 1
    ip_mid = inv_phi[mid]
    if np.sum(ip_mid**2) == 0: return None
    slope = np.sum(b_m1 * ip_mid) / np.sum(ip_mid**2)
    if slope <= 0: return None
    
    pred = 1 + slope * ip_mid
    ss_res = np.sum((boost[mid] - pred)**2)
    ss_tot = np.sum((boost[mid] - np.mean(boost[mid]))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    return {
        'name': d['name'], 'slope': slope, 'r2': r2,
        'intercept': intercept, 'GM_bar': GM_bar, 'M_bar': GM_bar/G_const,
        'sqrt_GM': np.sqrt(GM_bar), 'v_flat': vo[-1],
    }

# Process all
print("Analyzing all galaxies...")
all_results = []
for f in files:
    r = analyze_galaxy(read_sparc(f))
    if r is not None:
        all_results.append(r)
print(f"  {len(all_results)} galaxies with valid analysis")

# Two filter levels
strict = [r for r in all_results if r['r2'] > 0.9 and 0.5 <= r['intercept'] <= 1.5]
relaxed = [r for r in all_results if r['r2'] > 0.8 and 0.5 <= r['intercept'] <= 1.5]

print(f"  Strict  (R²>0.9, intcpt 0.5-1.5): {len(strict)}")
print(f"  Relaxed (R²>0.8, intcpt 0.5-1.5): {len(relaxed)}")

# --- PLOT ---
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

for ax_idx, (dataset, label, r2_thresh) in enumerate([
    (strict, 'R² > 0.9 (strict)', 0.9),
    (relaxed, 'R² > 0.8 (relaxed)', 0.8),
]):
    ax = axes[ax_idx]
    
    sqrt_gms = np.array([r['sqrt_GM'] for r in dataset])
    slopes = np.array([r['slope'] for r in dataset])
    r2s = np.array([r['r2'] for r in dataset])
    
    sc = ax.scatter(sqrt_gms, slopes, s=80, c=r2s, cmap='RdYlGn',
                    vmin=r2_thresh, vmax=1.0, zorder=5,
                    edgecolors='black', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label=f'R² (20-80%)', shrink=0.8)
    
    for r in dataset:
        ax.annotate(r['name'], (r['sqrt_GM']*1.01, r['slope']*1.01),
                    fontsize=5, alpha=0.7)
    
    # Fit line through origin: slope = C * sqrt(GM)
    C_lin = np.sum(slopes * sqrt_gms) / np.sum(sqrt_gms**2)
    x_line = np.linspace(0, sqrt_gms.max()*1.15, 200)
    ax.plot(x_line, C_lin * x_line, 'r--', lw=2, zorder=3,
            label=f'slope = {C_lin:.2e} · $\\sqrt{{GM}}$')
    
    # R² of this linear fit
    pred_lin = C_lin * sqrt_gms
    ss_res = np.sum((slopes - pred_lin)**2)
    ss_tot = np.sum((slopes - np.mean(slopes))**2)
    r2_lin = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    ax.text(0.05, 0.92, f'Linear through origin:\nslope = {C_lin:.2e} · √(GM)\nR² = {r2_lin:.3f}',
            transform=ax.transAxes, fontsize=10, fontweight='bold', color='red',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    
    ax.set_xlabel(r'$\sqrt{G \cdot M_{bar}}$   ($M_{bar}$ from baryonic data)', fontsize=13)
    ax.set_ylabel('slope', fontsize=13)
    ax.set_title(f'{len(dataset)} galaxies — {label}\nintercept ∈ [0.5, 1.5]',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-sqrt_gms.max()*0.03)
    ax.set_ylim(bottom=-slopes.max()*0.05)

plt.suptitle(
    'SPARC: slope vs $\\sqrt{GM_{bar}}$  —  boost = 1 + slope · (1/φ)\n'
    f'Y* = {Y_STAR}  |  M calculated from V²_bar · r_max / G',
    fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

outfile = 'sparc_filtered_dual.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight')
# Also save to outputs if running on Claude
if os.path.isdir('/mnt/user-data/outputs'):
    plt.savefig('/mnt/user-data/outputs/sparc_filtered_dual.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved {outfile}")
print(f"\n{'='*70}")
print(f"STRICT ({len(strict)} galaxies): C = {np.sum([r['slope']*r['sqrt_GM'] for r in strict])/np.sum([r['sqrt_GM']**2 for r in strict]):.4e}")
print(f"RELAXED ({len(relaxed)} galaxies): C = {np.sum([r['slope']*r['sqrt_GM'] for r in relaxed])/np.sum([r['sqrt_GM']**2 for r in relaxed]):.4e}")
print(f"\nFormel: a_tot = a_n · (1 + C · sqrt(GM) / phi)")
print(f"        boost = 1 + C · sqrt(GM) · (1/phi)")

