# NeurIPS US vs. China Research Output Analysis

This repository packages the scripts, datasets, and artifacts used to analyze NeurIPS accepted-paper affiliations (2021–2025) with a focus on comparing research output from the United States and China.

## Contents

```
.
├── README.md
├── requirements.txt
├── src/
│   └── run_analysis.py
├── data/
│   ├── raw/
│   │   └── neurips20XX_accepted.csv
│   ├── processed/
│   │   ├── affiliation_counts_full.csv
│   │   ├── affiliation_mapping_long.csv
│   │   ├── affiliation_normalization.json
│   │   ├── affiliation_region_mapping.json
│   │   ├── region_paper_counts.csv
│   │   ├── top_affiliations_by_year.csv
│   │   ├── unknown_region_papers.csv
│   │   └── us_china_collaboration.csv
│   └── reference/
│       └── world_universities_and_domains.json
└── figures/
    ├── region_share.png
    ├── region_share_stacked.png
    ├── region_trends.png
    ├── top_affiliation_bump.png
    └── us_china_collaboration.png
```

## Getting Started

1. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Reproduce the figures and tables**
   ```bash
   python src/run_analysis.py
   ```
   The script reads the raw NeurIPS CSV exports, normalizes and maps author affiliations to countries/regions, and regenerates both the processed datasets and figures in-place.

## Data Notes

- The NeurIPS accepted-paper exports (2021–2025) are sourced from openreview and are redistributed here (~34 MB total).
- `world_universities_and_domains.json` provides auxiliary affiliation → country mapping support.
- `unknown_region_papers.csv` enumerates papers that could not be mapped to a region (currently 302 across the five years).

## Outputs

Key derived artifacts include:
- `region_paper_counts.csv`: Unique-paper counts per region per year.
- `affiliation_counts_full.csv`: Paper counts per canonical affiliation across years.
- `top_affiliation_bump.png`: Trajectories for top affiliations from the US and China ecosystems.
- `region_share_stacked.png`: Marketing-ready stacked area chart showing regional share shifts.


## Reproducibility

The repository is self-contained; no files were removed from the original analysis pipeline. Re-running `run_analysis.py` will overwrite the processed data and figures with fresh outputs.
