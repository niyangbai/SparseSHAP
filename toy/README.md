# Comparing Custom SHAP Approximator with Original SHAP on Simulated Data

## Introduction

This repository demonstrates how to approximate SHAP (SHapley Additive exPlanations) values without relying on the original SHAP package. By simulating both tabular and sequential data with known true feature importances, we train machine learning models and compute SHAP values using both the original SHAP library and a custom SHAP approximator. The goal is to compare the effectiveness of the custom approximator against the original SHAP implementation and the true feature importances.

## Overview

### Objectives

- **Simulate Data**: Generate tabular and sequential datasets with known true feature importances using custom simulators.
- **Train Models**: Train an XGBoost regressor on tabular data and an LSTM neural network on sequential data.
- **Compute SHAP Values**:
  - **Original SHAP Package**: Use the SHAP library to compute SHAP values.
  - **Custom SHAP Approximator**: Compute SHAP values without the SHAP package, using custom approximation methods.
- **Compare and Analyze**: Evaluate and compare the SHAP values from both methods against the true feature importances, using visualization and statistical metrics.

### Structure

The repository includes:

- **Simulation Classes**: Abstract base classes and implementations for simulating tabular and sequential data.
- **Custom SHAP Approximators**: Abstract base classes and implementations for approximating SHAP values for both tabular and sequential models.
- **Jupyter Notebook**: A notebook that runs the simulations, trains the models, computes SHAP values using both methods, and compares the results through plots and correlation analyses.

## Conclusion

By approximating SHAP values without relying on the SHAP package, this project provides insights into the interpretability of machine learning models and the effectiveness of custom explanation methods. The comparisons show that the custom SHAP approximator can produce results similar to the original SHAP implementation, offering an alternative approach for model explanation that can be further developed and customized.
