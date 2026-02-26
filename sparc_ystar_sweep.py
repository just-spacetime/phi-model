import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob

G_CONST = 6.67430e-11
M_SUN = 1.989e30
KPC_TO_M = 3.086e19
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

# Pre-load all galaxies
print(f"Loading {len(files)} galaxies...")
all_gals = [read_sparc(f) for f in files if len(read_sparc(f)['Rad']) >= 5]
# Re-read properly
all_gals = []
for f in files:
    d = read_sparc(f)
    if len(d['Rad']) >= 5:
        all_gals.append(d)
print(f"  {len(all_gals)} galaxies loaded")

# Sweep Y*
ystar_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]

sweep_results = []

for Y_STAR in ystar_values:
    n_total = 0
    wins_phi = 0
    wins_mond = 0
    rmse_phi_list = []
    rmse_mond_list = []
    
    for d in all_gals:
        rad = d['Rad']; vobs = d['Vobs']; errv = d['errV']
        v2_gas = d['Vgas'] * np.abs(d['Vgas'])
        v2_disk = Y_STAR * d['Vdisk'] * np.abs(d['Vdisk'])
        v2_bul = Y_STAR * d['Vbul'] * np.abs(d['Vbul'])
        v2_bar = v2_gas + v2_disk + v2_bul
        v_bar = np.sqrt(np.maximum(v2_bar, 0))
        
        mask = (v2_bar > 0) & (vobs > 10) & (rad > 0.1) & (v_bar > 5)
        if np.sum(mask) < 5: continue
        
        r = rad[mask]; vo = vobs[mask]; vb = v_bar[mask]
        
        try:
            phi, an = compute_potential(r, v2_bar[mask])
        except:
            continue
        
        r_m = r * KPC_TO_M
        
        # MOND
        a_mond = 0.5 * (an + np.sqrt(an**2 + 4*an*A0_MOND))
        v_mond = np.sqrt(a_mond * r_m) / 1000
        
        # PHI model
        boost_phi = np.maximum(1.0 + C_PHI / np.sqrt(an), 0.1)
        v_phi = np.sqrt(an * boost_phi * r_m) / 1000
        
        rmse_mond = np.sqrt(np.mean((v_mond - vo)**2))
        rmse_phi = np.sqrt(np.mean((v_phi - vo)**2))
        
        n_total += 1
        rmse_phi_list.append(rmse_phi)
        rmse_mond_list.append(rmse_mond)
        if rmse_phi < rmse_mond:
            wins_phi += 1
        else:
            wins_mond += 1
    
    pct_phi = 100*wins_phi/n_total if n_total > 0 else 0
    mean_rmse_phi = np.mean(rmse_phi_list) if rmse_phi_list else 0
    mean_rmse_mond = np.mean(rmse_mond_list) if rmse_mond_list else 0
    med_rmse_phi = np.median(rmse_phi_list) if rmse_phi_list else 0
    med_rmse_mond = np.median(rmse_mond_list) if rmse_mond_list else 0
    
    sweep_results.append({
        'ystar': Y_STAR, 'n': n_total,
        'wins_phi': wins_phi, 'wins_mond': wins_mond, 'pct_phi': pct_phi,
        'mean_rmse_phi': mean_rmse_phi, 'mean_rmse_mond': mean_rmse_mond,
        'med_rmse_phi': med_rmse_phi, 'med_rmse_mond': med_rmse_mond,
    })
    
    print(f"  Y*={Y_STAR:.2f}: n={n_total}, φ wins {wins_phi} ({pct_phi:.0f}%), "
          f"RMSE φ={mean_rmse_phi:.1f} M={mean_rmse_mond:.1f}")

# === TABLE ===
print(f"\n{'='*80}")
print(f"  Y* SWEEP: φ-Model (C={C_PHI:.1e}) vs MOND (a0={A0_MOND:.1e})")
print(f"{'='*80}")
print(f"  {'Y*':>5s}  {'N':>4s}  {'φ wins':>7s}  {'MOND wins':>9s}  {'φ%':>5s}  {'RMSE_φ':>7s}  {'RMSE_M':>7s}  {'med_φ':>7s}  {'med_M':>7s}  {'Ratio':>6s}")
print(f"  {'-'*75}")
for r in sweep_results:
    ratio = r['med_rmse_phi']/r['med_rmse_mond'] if r['med_rmse_mond'] > 0 else 99
    marker = " ←" if r['pct_phi'] > 45 else ""
    print(f"  {r['ystar']:5.2f}  {r['n']:4d}  {r['wins_phi']:7d}  {r['wins_mond']:9d}  {r['pct_phi']:5.0f}%  "
          f"{r['mean_rmse_phi']:7.1f}  {r['mean_rmse_mond']:7.1f}  "
          f"{r['med_rmse_phi']:7.1f}  {r['med_rmse_mond']:7.1f}  {ratio:6.2f}{marker}")

# === PLOT ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ys = [r['ystar'] for r in sweep_results]
phi_pct = [r['pct_phi'] for r in sweep_results]
mean_phi = [r['mean_rmse_phi'] for r in sweep_results]
mean_mond = [r['mean_rmse_mond'] for r in sweep_results]
med_phi = [r['med_rmse_phi'] for r in sweep_results]
med_mond = [r['med_rmse_mond'] for r in sweep_results]

# Top left: Win percentage
ax = axes[0, 0]
ax.plot(ys, phi_pct, 'bo-', lw=2.5, ms=8, label='φ-Model win %')
ax.axhline(50, color='red', ls='--', lw=1.5, label='50% (equal)')
ax.axhline(25, color='gray', ls=':', alpha=0.5)
ax.fill_between(ys, 50, 100, color='blue', alpha=0.05)
ax.fill_between(ys, 0, 50, color='orange', alpha=0.05)
ax.set_xlabel('Y* (stellar mass-to-light ratio)', fontsize=13)
ax.set_ylabel('φ-Model win percentage [%]', fontsize=13)
ax.set_title('How often does φ-Model beat MOND?', fontsize=13, fontweight='bold')
ax.set_ylim(0, 65)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.text(0.3, 0.85, 'φ better', transform=ax.transAxes, fontsize=12, color='blue', alpha=0.5)
ax.text(0.3, 0.15, 'MOND better', transform=ax.transAxes, fontsize=12, color='orange', alpha=0.5)
# Mark DM-preferred and MOND-preferred Y* ranges
ax.axvspan(0.2, 0.35, color='red', alpha=0.08, label='DM preferred')
ax.axvspan(0.5, 0.7, color='green', alpha=0.08, label='MOND preferred')
ax.legend(fontsize=8, loc='upper left')

# Top right: Mean RMSE
ax = axes[0, 1]
ax.plot(ys, mean_phi, 'b-o', lw=2, ms=6, label='φ-Model mean RMSE')
ax.plot(ys, mean_mond, 'orange', ls='-', marker='s', lw=2, ms=6, label='MOND mean RMSE')
ax.set_xlabel('Y*', fontsize=13)
ax.set_ylabel('Mean RMSE [km/s]', fontsize=13)
ax.set_title('Mean RMSE vs Y*', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.axvspan(0.2, 0.35, color='red', alpha=0.08)
ax.axvspan(0.5, 0.7, color='green', alpha=0.08)

# Bottom left: Median RMSE
ax = axes[1, 0]
ax.plot(ys, med_phi, 'b-o', lw=2, ms=6, label='φ-Model median RMSE')
ax.plot(ys, med_mond, 'orange', ls='-', marker='s', lw=2, ms=6, label='MOND median RMSE')
ax.set_xlabel('Y*', fontsize=13)
ax.set_ylabel('Median RMSE [km/s]', fontsize=13)
ax.set_title('Median RMSE vs Y*', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.axvspan(0.2, 0.35, color='red', alpha=0.08)
ax.axvspan(0.5, 0.7, color='green', alpha=0.08)

# Bottom right: RMSE ratio
ax = axes[1, 1]
ratio_mean = [p/m if m > 0 else 99 for p, m in zip(mean_phi, mean_mond)]
ratio_med = [p/m if m > 0 else 99 for p, m in zip(med_phi, med_mond)]
ax.plot(ys, ratio_mean, 'r-o', lw=2, ms=6, label='Mean RMSE ratio (φ/MOND)')
ax.plot(ys, ratio_med, 'b-s', lw=2, ms=6, label='Median RMSE ratio (φ/MOND)')
ax.axhline(1.0, color='black', ls='--', lw=1.5, label='equal performance')
ax.set_xlabel('Y*', fontsize=13)
ax.set_ylabel('RMSE(φ) / RMSE(MOND)', fontsize=13)
ax.set_title('Performance ratio vs Y*\n<1 = φ wins, >1 = MOND wins', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.axvspan(0.2, 0.35, color='red', alpha=0.08)
ax.axvspan(0.5, 0.7, color='green', alpha=0.08)
ax.set_ylim(0.8, 1.5)

plt.suptitle(
    f'Y* Sweep: φ-Model vs MOND across SPARC\n'
    f'φ: $a_{{tot}} = a_n(1 + C/\\sqrt{{a_n}})$, C={C_PHI:.1e}  |  '
    f'MOND: simple μ, $a_0$={A0_MOND:.1e}\n'
    f'Red band = DM-preferred Y*, Green band = MOND-preferred Y*',
    fontsize=12, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('sparc_ystar_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved sparc_ystar_sweep.png")

