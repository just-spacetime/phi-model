import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob

G_CONST = 6.67430e-11
M_SUN = 1.989e30
KPC_TO_M = 3.086e19
Y_STAR = 0.5
C_PHI = 0.9e-5

SPARC_DIR = './sparc_data'
files = sorted(glob.glob(os.path.join(SPARC_DIR, '*_rotmod.dat')))

def read_sparc(filepath):
    rad, vobs, errv, vgas, vdisk, vbul = [], [], [], [], [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
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

# ============================================================
# ANALYTICAL CHECK of Tully-Fisher from our formula
# ============================================================
print("="*70)
print("ANALYTICAL TULLY-FISHER CHECK")
print("="*70)
print("""
Our formula: a_tot = a_n * (1 + C / sqrt(a_n))

For large r in the outer regions where a_n is small:
  The correction term C / sqrt(a_n) becomes significant.

For a point mass: a_n = GM/r²

So: a_tot = GM/r² * (1 + C / sqrt(GM/r²))
         = GM/r² * (1 + C * r / sqrt(GM))
         = GM/r² + C * sqrt(GM) / r

For large r (low acceleration regime), the second term dominates:
  a_tot ≈ C * sqrt(GM) / r

Circular velocity: V² = a_tot * r = C * sqrt(GM)

Therefore: V² = C * sqrt(GM)
           V⁴ = C² * GM

This gives: V⁴ ∝ M  (with slope C² * G)

This IS the baryonic Tully-Fisher relation!

Predicted slope: V⁴ = C² * G * M
With C = 0.9e-5: V⁴ = (0.9e-5)² * 6.674e-11 * M = 5.406e-21 * M

MOND predicts: V⁴ = a₀ * G * M = 1.2e-10 * 6.674e-11 * M = 8.009e-21 * M
""")

# ============================================================
# EMPIRICAL CHECK on SPARC data
# ============================================================
print("="*70)
print("EMPIRICAL CHECK ON SPARC")
print("="*70)

v_flats = []
m_bars = []
names_out = []

for f in files:
    d = read_sparc(f)
    if len(d['Rad']) < 5: continue
    
    rad = d['Rad']; vobs = d['Vobs']
    v2_gas = d['Vgas']*np.abs(d['Vgas'])
    v2_disk = Y_STAR*d['Vdisk']*np.abs(d['Vdisk'])
    v2_bul = Y_STAR*d['Vbul']*np.abs(d['Vbul'])
    v2_bar = v2_gas + v2_disk + v2_bul
    
    mask = (vobs > 10) & (rad > 0.1) & (v2_bar > 0)
    if np.sum(mask) < 5: continue
    
    # V_flat: mean of outer 30% of observed velocities
    r_m = rad[mask]; vo = vobs[mask]
    n_outer = max(2, len(vo)//3)
    v_flat = np.mean(vo[-n_outer:])
    
    # M_bar from V²_bar at last point
    r_last = r_m[-1] * KPC_TO_M
    v2_last = np.abs(v2_bar[mask][-1]) * 1e6  # (m/s)²
    GM = v2_last * r_last
    M_bar = GM / G_CONST
    
    if M_bar > 0 and v_flat > 20:
        v_flats.append(v_flat)
        m_bars.append(M_bar)
        names_out.append(d['name'])

v_flats = np.array(v_flats)
m_bars = np.array(m_bars)

print(f"\n{len(v_flats)} galaxies with valid V_flat and M_bar")

# Tully-Fisher: log(V_flat) vs log(M_bar)
log_v = np.log10(v_flats)
log_m = np.log10(m_bars / M_SUN)

# Fit: log(V) = a * log(M) + b  →  V ∝ M^a
coeffs_tf = np.polyfit(log_m, log_v, 1)
a_tf = coeffs_tf[0]
print(f"\nEmpirical Tully-Fisher: log(V_flat) = {a_tf:.4f} * log(M/M☉) + {coeffs_tf[1]:.4f}")
print(f"  → V ∝ M^{a_tf:.4f}")
print(f"  → V⁴ ∝ M^{4*a_tf:.4f}")
print(f"  Expected from our model: V⁴ ∝ M^1.0 → V ∝ M^0.25")
print(f"  Expected from MOND:      V⁴ ∝ M^1.0 → V ∝ M^0.25")
print(f"  Observed (McGaugh 2012):  V⁴ ∝ M^~1.0 → V ∝ M^~0.25")

# R² of TF fit
pred_log_v = np.polyval(coeffs_tf, log_m)
ss_res = np.sum((log_v - pred_log_v)**2)
ss_tot = np.sum((log_v - np.mean(log_v))**2)
r2_tf = 1 - ss_res/ss_tot
print(f"\n  R² of TF fit: {r2_tf:.4f}")

# Now check: does V⁴ = C² * G * M hold?
v4_obs = (v_flats * 1000)**4  # (m/s)⁴
gm_bar = G_CONST * m_bars     # G*M in SI

# Fit V⁴ = slope * GM
slope_v4 = np.sum(v4_obs * gm_bar) / np.sum(gm_bar**2)
print(f"\n  Direct fit: V⁴ = {slope_v4:.4e} * GM")
print(f"  Our prediction: V⁴ = C² * GM = {C_PHI**2:.4e} * GM")
print(f"  MOND prediction: V⁴ = a₀ * GM = {1.2e-10:.4e} * GM")
print(f"  Ratio observed/predicted(ours): {slope_v4/C_PHI**2:.2f}")
print(f"  Ratio observed/predicted(MOND): {slope_v4/1.2e-10:.2f}")

# Also compute what C would need to be
C_from_tf = np.sqrt(slope_v4)
print(f"\n  C implied by TF data: {C_from_tf:.4e}")
print(f"  C from rotation curve fits: {C_PHI:.4e}")
print(f"  Ratio: {C_from_tf/C_PHI:.2f}")

# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Plot 1: Standard Tully-Fisher (log-log)
ax = axes[0]
ax.scatter(log_m, log_v, s=20, c='darkblue', alpha=0.7, zorder=5, edgecolors='black', linewidths=0.3)
m_line = np.linspace(log_m.min(), log_m.max(), 100)
ax.plot(m_line, np.polyval(coeffs_tf, m_line), 'r-', lw=2,
        label=f'Fit: V ∝ M$^{{{a_tf:.3f}}}$ (R²={r2_tf:.3f})')
ax.plot(m_line, 0.25*m_line + np.log10(C_PHI*1e-3/G_CONST**0.25*1e3), 'g--', lw=1.5, alpha=0.5)
ax.set_xlabel('log₁₀(M_bar / M☉)', fontsize=13)
ax.set_ylabel('log₁₀(V_flat / km/s)', fontsize=13)
ax.set_title('Baryonic Tully-Fisher Relation\n(SPARC data)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Plot 2: V⁴ vs GM (linear)
ax = axes[1]
ax.scatter(gm_bar, v4_obs, s=20, c='darkred', alpha=0.7, zorder=5, edgecolors='black', linewidths=0.3)
gm_line = np.linspace(0, gm_bar.max()*1.1, 100)
ax.plot(gm_line, slope_v4 * gm_line, 'r-', lw=2, label=f'Fit: V⁴ = {slope_v4:.2e}·GM')
ax.plot(gm_line, C_PHI**2 * gm_line, 'b--', lw=2, label=f'φ-Model: V⁴ = C²·GM = {C_PHI**2:.1e}·GM')
ax.plot(gm_line, 1.2e-10 * gm_line, 'orange', ls='--', lw=2, label=f'MOND: V⁴ = a₀·GM = 1.2e-10·GM')
ax.set_xlabel('GM [m³/s²]', fontsize=12)
ax.set_ylabel('V⁴_flat [(m/s)⁴]', fontsize=12)
ax.set_title('V⁴ vs GM — Tully-Fisher\nlinear scale', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Plot 3: V⁴ vs GM (log-log)
ax = axes[2]
ax.scatter(gm_bar, v4_obs, s=20, c='darkgreen', alpha=0.7, zorder=5, edgecolors='black', linewidths=0.3)
ax.plot(gm_line, slope_v4 * gm_line, 'r-', lw=2, label=f'Fit: slope={slope_v4:.2e}')
ax.plot(gm_line, C_PHI**2 * gm_line, 'b--', lw=2, label=f'C²={C_PHI**2:.1e}')
ax.plot(gm_line, 1.2e-10 * gm_line, 'orange', ls='--', lw=2, label=f'a₀={1.2e-10:.1e}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('GM [m³/s²]', fontsize=12)
ax.set_ylabel('V⁴_flat [(m/s)⁴]', fontsize=12)
ax.set_title('V⁴ vs GM — log scale', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Tully-Fisher from φ-Model:  V⁴ = C²·GM  vs  MOND: V⁴ = a₀·GM\n'
    f'Empirical: V ∝ M$^{{{a_tf:.3f}}}$, i.e. V⁴ ∝ M$^{{{4*a_tf:.2f}}}$  |  '
    f'C²={C_PHI**2:.1e}, a₀={1.2e-10:.1e}',
    fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('sparc_tully_fisher.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved sparc_tully_fisher.png")

