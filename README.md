# Hierarchical IBD Screening Calculator

Interactive tool for designing and evaluating multi-stage IBD screening pipelines.

**Live demo:** https://sazonovsgrouppredict.github.io/hierarchical/

## Features

- Drag-and-drop pipeline builder with customizable screening blocks (PRS, blood biomarkers, fecal calprotectin, imaging, etc.)
- Real-time calculation of sensitivity, specificity, PPV, and cost
- Monte Carlo simulation with confidence intervals
- Correlation modeling between screening layers
- Single-test equivalent AUC comparison
- Cost breakdown analysis

## Model

Uses a binormal signal detection model where cases and controls follow normal distributions separated by d' (derived from AUC). For each layer, a threshold is found that selects the top q% of the current cohort, yielding layer-specific sensitivity and false positive rates.

## Usage

1. Add screening blocks to the pipeline
2. Adjust AUC, cohort selection %, and cost for each block
3. View results in real-time on the right panel
4. Open the Dashboard for detailed visualizations
5. Use Config to save/load pipeline setups as JSON

## Citation

Sazonovs Lab, PREDICT, AAU Copenhagen
