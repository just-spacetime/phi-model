# phi-model
Galaxy rotation curves from baryonic potential geometry — no dark matter, one constant

## Paper

The full paper is available at:  
**https://doi.org/10.5281/zenodo.18792263**

> Nicke, M. (2026). *The Information Is in the Potential: Galaxy Rotation Curves from Baryonic Field Geometry.* Preprint, February 2026.

## Summary

This repository contains the analysis code for a single-parameter model of galaxy rotation curves based entirely on the baryonic gravitational potential — no dark matter, no per-galaxy free parameters.

The model:

```
a_total = a_N · (1 + C / √a_N)
```

where `a_N` is the Newtonian baryonic acceleration and `C ≈ 0.9 × 10⁻⁵ m^(1/2)/s` is a single universal constant.

Tested against **175 SPARC galaxies**, the model performs on par with MOND — with a marginal edge (54% vs 46% win rate, median RMSE ratio 0.97) — using one constant versus MOND's one constant.

## Data

SPARC rotation curve data is **not included** in this repository. Download it from the official sources:

- SPARC website: http://astroweb.case.edu/SPARC/
- Zenodo mirror: https://doi.org/10.5281/zenodo.16284118

Download the `*_rotmod.dat` files and place them in a folder called `sparc_data/` next to the scripts.

## Scripts

| Script | Description |
|--------|-------------|
| `sparc_all_175_optimized.py` | Main dashboard: 175 galaxies × 4 panels, optimizes C, compares with MOND |
| `sparc_filtered_dual.py` | Slope–mass relation for quality-filtered subsample |
| `sparc_phi_vs_mond.py` | Head-to-head statistical comparison φ-Model vs MOND |
| `sparc_ystar_sweep.py` | Sensitivity analysis: performance vs stellar mass-to-light ratio Υ* |
| `sparc_tully_fisher.py` | Baryonic Tully-Fisher relation |

## Requirements

```
python3
numpy
matplotlib
scipy
```

Install with:
```bash
pip install numpy matplotlib scipy
```

## Reproduce

```bash
git clone https://github.com/just-spacetime/phi-model
cd phi-model
# Download SPARC data into sparc_data/
python sparc_all_175_optimized.py
```

## License

- **Code**: MIT License
- **Paper**: © 2026 Markus Nicke. All rights reserved.

## Acknowledgments

SPARC database: Lelli, McGaugh & Schombert (2016), AJ, 152, 157.
