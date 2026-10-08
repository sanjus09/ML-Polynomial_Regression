# Machine Learning Assignment 1: Polynomial Regression

This repository contains the end-to-end implementation for predicting continuous target variables in geothermal energy datasets using Polynomial Regression with L2 Regularization (Ridge Regression). The core mathematical models and cross-validation pipelines were built from scratch using NumPy.

## Repository Structure

* **`data/`**: Contains the assigned training and testing CSV datasets.
* **`model1.ipynb`**: 5-Fold Cross-Validation and training pipeline for Phase 1 (Steam Turbine Optimization).
* **`model2.ipynb`**: 5-Fold Cross-Validation and training pipeline for Phase 2 (Subterranean Thermal Reservoir Mapping).
* **`prediction_files/`**: Contains the final exported CSV predictions generated for submission.
* **`Test_files/`**: Experimental scripts and rough tests (can be safely ignored during evaluation).

## Methodology Highlights

* **Custom Implementation**: Matrix operations for Ridge Regression and evaluation metrics (MSE, R-squared) are implemented purely in `numpy` to demonstrate mathematical understanding.
* **Model Selection**: 5-Fold Cross-Validation combined with a grid search over polynomial degrees and lambda penalties to prevent overfitting.
* **Feature Engineering**: `scikit-learn` was used strictly for generating polynomial combinations. 

## How to Run

1. Ensure the required Python dependencies are installed:
   ```bash
   pip install numpy pandas scikit-learn jupyter