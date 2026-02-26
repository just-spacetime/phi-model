import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob

G_CONST = 6.67430e-11
M_SUN = 1.989e30
KPC_TO_M = 3.086e19
Y_STAR = 0.5
A0_MOND = 1.2e-10
C_PHI = 0.9e-5

SPARC_DIR = './sparc_data'
files = sorted(glob.glob(os.path.join(SPARC_DIR, '*_rotmod.dat')))

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

results = []

for filepath in files:
    d = read_sparc(filepath)
    if len(d['Rad']) < 5: continue
    
    rad = d['Rad']; vobs = d['Vobs']; errv = d['errV']
    v2_gas = d['Vgas'] * np.abs(d['Vgas'])
    v2_disk = Y_STAR * d['Vdisk'] * np.abs(d['Vdisk'])
    v2_bul = Y_STAR * d['Vbul'] * np.abs(d['Vbul'])
    v2_bar = v2_gas + v2_disk + v2_bul
    v_bar = np.sqrt(np.maximum(v2_bar, 0))
    
    mask = (v2_bar > 0) & (vobs > 10) & (rad > 0.1) & (v_bar > 5)
    if np.sum(mask) < 5: continue
    
    r = rad[mask]; vo = vobs[mask]; vb = v_bar[mask]; ev = errv[mask]
    
    phi, an = compute_potential(r, v2_bar[mask])
    
    # (1/phi)' mean for mass proxy
    inv_phi = 1.0 / phi
    d_inv_phi = np.gradient(inv_phi, r * KPC_TO_M)
    n4 = max(1, len(d_inv_phi)//4)
    fp_mean = np.mean(np.abs(d_inv_phi[n4:]))
    if fp_mean <= 0: continue
    inv_sqrt_fp = 1.0 / np.sqrt(fp_mean)
    
    r_m = r * KPC_TO_M
    
    # MOND velocity
    a_mond = 0.5 * (an + np.sqrt(an**2 + 4*an*A0_MOND))
    v_mond = np.sqrt(a_mond * r_m) / 1000
    
    # PHI model velocity
    boost_phi = 1.0 + C_PHI / np.sqrt(an)
    boost_phi = np.maximum(boost_phi, 0.1)
    v_phi = np.sqrt(an * boost_phi * r_m) / 1000
    
    # RMSE
    rmse_mond = np.sqrt(np.mean((v_mond - vo)**2))
    rmse_phi = np.sqrt(np.mean((v_phi - vo)**2))
    
    # Weighted RMSE (by 1/errv, skip errv=0)
    w = np.where(ev > 0, 1.0/ev, 0)
    if np.sum(w) > 0:
        wrmse_mond = np.sqrt(np.sum(w*(v_mond-vo)**2)/np.sum(w))
        wrmse_phi = np.sqrt(np.sum(w*(v_phi-vo)**2)/np.sum(w))
    else:
        wrmse_mond = rmse_mond
        wrmse_phi = rmse_phi
    
    # Chi² (reduced)
    mask_err = ev > 0
    if np.sum(mask_err) > 2:
        chi2_mond = np.sum(((v_mond[mask_err]-vo[mask_err])/ev[mask_err])**2) / np.sum(mask_err)
        chi2_phi = np.sum(((v_phi[mask_err]-vo[mask_err])/ev[mask_err])**2) / np.sum(mask_err)
    else:
        chi2_mond = rmse_mond**2
        chi2_phi = rmse_phi**2
    
    # Mass
    GM = an[-1] * r_m[-1]**2
    M_bar = GM / G_CONST
    
    results.append({
        'name': d['name'],
        'rmse_mond': rmse_mond, 'rmse_phi': rmse_phi,
        'wrmse_mond': wrmse_mond, 'wrmse_phi': wrmse_phi,
        'chi2_mond': chi2_mond, 'chi2_phi': chi2_phi,
        'M_bar': M_bar, 'n_pts': np.sum(mask),
        'winner_rmse': 'phi' if rmse_phi < rmse_mond else 'mond',
        'winner_chi2': 'phi' if chi2_phi < chi2_mond else 'mond',
    })

print(f"Analyzed {len(results)} galaxies\n")

# === STATISTICS ===
n = len(results)
wins_phi_rmse = sum(1 for r in results if r['winner_rmse'] == 'phi')
wins_mond_rmse = n - wins_phi_rmse
wins_phi_chi2 = sum(1 for r in results if r['winner_chi2'] == 'phi')
wins_mond_chi2 = n - wins_phi_chi2

rmse_mond_all = np.array([r['rmse_mond'] for r in results])
rmse_phi_all = np.array([r['rmse_phi'] for r in results])
chi2_mond_all = np.array([r['chi2_mond'] for r in results])
chi2_phi_all = np.array([r['chi2_phi'] for r in results])
masses = np.array([r['M_bar'] for r in results])

print("="*70)
print(f"  SCOREBOARD: φ-Model vs MOND  ({n} galaxies)")
print("="*70)
print(f"\n  By RMSE:")
print(f"    φ-Model wins:  {wins_phi_rmse:3d} ({100*wins_phi_rmse/n:.0f}%)")
print(f"    MOND wins:     {wins_mond_rmse:3d} ({100*wins_mond_rmse/n:.0f}%)")
print(f"\n  By reduced χ²:")
print(f"    φ-Model wins:  {wins_phi_chi2:3d} ({100*wins_phi_chi2/n:.0f}%)")
print(f"    MOND wins:     {wins_mond_chi2:3d} ({100*wins_mond_chi2/n:.0f}%)")
print(f"\n  Mean RMSE:")
print(f"    MOND:     {np.mean(rmse_mond_all):6.1f} km/s")
print(f"    φ-Model:  {np.mean(rmse_phi_all):6.1f} km/s")
print(f"\n  Median RMSE:")
print(f"    MOND:     {np.median(rmse_mond_all):6.1f} km/s")
print(f"    φ-Model:  {np.median(rmse_phi_all):6.1f} km/s")
print(f"\n  Mean reduced χ²:")
print(f"    MOND:     {np.mean(chi2_mond_all):6.2f}")
print(f"    φ-Model:  {np.mean(chi2_phi_all):6.2f}")
print(f"\n  Median reduced χ²:")
print(f"    MOND:     {np.median(chi2_mond_all):6.2f}")
print(f"    φ-Model:  {np.median(chi2_phi_all):6.2f}")

# By mass bins
print(f"\n  By mass bin (RMSE wins):")
mass_bins = [(0, 1e9, '<1e9'), (1e9, 1e10, '1e9-1e10'), (1e10, 1e11, '1e10-1e11'), (1e11, 1e13, '>1e11')]
for lo, hi, label in mass_bins:
    in_bin = [r for r in results if lo*M_SUN <= r['M_bar'] < hi*M_SUN]
    if len(in_bin) == 0: continue
    phi_w = sum(1 for r in in_bin if r['winner_rmse'] == 'phi')
    mond_w = len(in_bin) - phi_w
    mean_rmse_m = np.mean([r['rmse_mond'] for r in in_bin])
    mean_rmse_p = np.mean([r['rmse_phi'] for r in in_bin])
    print(f"    {label:12s}: {len(in_bin):3d} gal  |  φ wins {phi_w:2d}, MOND wins {mond_w:2d}  |  RMSE: φ={mean_rmse_p:.1f}, M={mean_rmse_m:.1f}")

# === DETAILED TABLE ===
print(f"\n\n{'Name':20s} {'M[M☉]':>10s} {'RMSE_φ':>8s} {'RMSE_M':>8s} {'χ²_φ':>7s} {'χ²_M':>7s} {'Win':>5s}")
print("-"*70)
results.sort(key=lambda r: r['M_bar'])
for r in results:
    w = "φ" if r['winner_rmse'] == 'phi' else "M"
    print(f"{r['name']:20s} {r['M_bar']/M_SUN:10.1e} {r['rmse_phi']:8.1f} {r['rmse_mond']:8.1f} {r['chi2_phi']:7.1f} {r['chi2_mond']:7.1f} {w:>5s}")

# === PLOT ===
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Top left: RMSE comparison scatter
ax = axes[0, 0]
max_rmse = max(rmse_mond_all.max(), rmse_phi_all.max()) * 1.1
ax.scatter(rmse_mond_all, rmse_phi_all, s=30, c=np.log10(masses/M_SUN),
           cmap='viridis', zorder=5, edgecolors='black', linewidths=0.3)
ax.plot([0, max_rmse], [0, max_rmse], 'r--', lw=1.5, label='equal')
ax.set_xlabel('RMSE MOND [km/s]', fontsize=12)
ax.set_ylabel('RMSE φ-Model [km/s]', fontsize=12)
ax.set_title(f'RMSE: φ-Model vs MOND\nφ wins {wins_phi_rmse}/{n} = {100*wins_phi_rmse/n:.0f}%',
             fontsize=12, fontweight='bold')
ax.set_xlim(0, min(max_rmse, 80)); ax.set_ylim(0, min(max_rmse, 80))
ax.legend(); ax.grid(True, alpha=0.3)
ax.fill_between([0, max_rmse], [0, 0], [0, max_rmse], color='blue', alpha=0.05)
ax.fill_between([0, max_rmse], [0, max_rmse], [max_rmse, max_rmse], color='red', alpha=0.05)
ax.text(0.7, 0.15, 'φ better', transform=ax.transAxes, fontsize=11, color='blue', alpha=0.5)
ax.text(0.15, 0.85, 'MOND better', transform=ax.transAxes, fontsize=11, color='red', alpha=0.5)

# Top right: histogram of RMSE ratio
ax = axes[0, 1]
ratio = rmse_phi_all / rmse_mond_all
ax.hist(ratio, bins=30, range=(0, 3), color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(1.0, color='red', lw=2, ls='--', label='equal')
ax.axvline(np.median(ratio), color='blue', lw=2, label=f'median = {np.median(ratio):.2f}')
ax.set_xlabel('RMSE(φ) / RMSE(MOND)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'RMSE Ratio Distribution\n<1 = φ better, >1 = MOND better', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Bottom left: wins by mass
ax = axes[1, 0]
log_masses = np.log10(masses / M_SUN)
bins_edge = np.arange(7, 12.5, 0.5)
phi_wins_hist = []
mond_wins_hist = []
bin_centers = []
for j in range(len(bins_edge)-1):
    lo, hi = bins_edge[j], bins_edge[j+1]
    in_bin = [(r, lm) for r, lm in zip(results, log_masses) if lo <= lm < hi]
    if len(in_bin) == 0:
        phi_wins_hist.append(0); mond_wins_hist.append(0)
    else:
        pw = sum(1 for r, _ in in_bin if r['winner_rmse'] == 'phi')
        phi_wins_hist.append(pw); mond_wins_hist.append(len(in_bin) - pw)
    bin_centers.append((lo+hi)/2)

x_pos = np.arange(len(bin_centers))
w = 0.35
ax.bar(x_pos - w/2, phi_wins_hist, w, color='blue', alpha=0.7, label='φ-Model wins')
ax.bar(x_pos + w/2, mond_wins_hist, w, color='orange', alpha=0.7, label='MOND wins')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{c:.1f}' for c in bin_centers], fontsize=8)
ax.set_xlabel('log₁₀(M_bar / M☉)', fontsize=12)
ax.set_ylabel('Number of galaxies', fontsize=12)
ax.set_title('Wins by mass bin', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Bottom right: RMSE difference vs mass
ax = axes[1, 1]
rmse_diff = rmse_mond_all - rmse_phi_all  # positive = phi better
sc = ax.scatter(masses/M_SUN, rmse_diff, s=30, c=np.where(rmse_diff > 0, 'blue', 'orange'),
                zorder=5, edgecolors='black', linewidths=0.3)
ax.axhline(0, color='red', lw=1.5, ls='--')
ax.set_xscale('log')
ax.set_xlabel('M_bar [M☉]', fontsize=12)
ax.set_ylabel('RMSE(MOND) - RMSE(φ)  [km/s]', fontsize=12)
ax.set_title('RMSE difference vs mass\n>0 = φ better (blue), <0 = MOND better (orange)',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle(
    f'SPARC {n} Galaxies: φ-Model vs MOND\n'
    f'φ-Model: $a_{{tot}} = a_n(1 + C/\\sqrt{{a_n}})$, C={C_PHI:.1e}  |  '
    f'MOND: simple μ, $a_0$={A0_MOND:.1e}',
    fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('sparc_phi_vs_mond.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved sparc_phi_vs_mond.png")

