# 📊 Credit Risk Model Audit  
## Fairness, Robustness & Stability Analysis

---

## Overview

This project presents an **end-to-end audit of a credit risk classification model**, focusing not only on predictive performance, but also on **fairness across demographic groups**, **group-level disparities**, and **model robustness under data perturbations**.

Instead of optimizing for accuracy alone, the goal is to answer **production-level questions**, such as:

- Does the model behave consistently across different demographic groups?
- Are some groups systematically under- or over-penalized?
- Is the model stable when the input data slightly changes?

This reflects real-world responsibilities of a **Data Scientist working with risk-sensitive models**, especially in regulated domains.

---

## Problem Statement

Given historical credit card customer data, the task is to **predict default risk** as a binary classification problem:

- **DEFAULT = 1** → customer failed to pay (default)
- **DEFAULT = 0** → customer paid normally

Beyond prediction, the project emphasizes **risk-aware evaluation and auditing**.

---

## Dataset

- **Source:** UCI Credit Card Default Dataset  
- **Size:** ~30,000 records  
- **Target variable:** `DEFAULT`

### Features include:
- Credit limit  
- Demographics (age, sex, education, marital status)  
- Payment history  
- Billing and payment amounts  

An additional categorical feature, **`AGE_BIN`**, was created to enable **age-based group auditing**.

---

## Methodology

### 1. Data Preparation
- Removed duplicated headers and non-informative columns  
- Converted all numerical features to proper numeric types  
- Created categorical age groups (`AGE_BIN`)  
- Applied a stratified train/test split to preserve default rate distribution  

---

### 2. Modeling Pipeline

A **fully reproducible pipeline** was built using **scikit-learn**, with all preprocessing steps included to avoid data leakage.

#### Numerical features
- Median imputation  
- Standard scaling  

#### Categorical features
- Most-frequent imputation  
- One-hot encoding  

#### Model
- **Logistic Regression**
  - Baseline model  
  - Interpretable  
  - Industry-standard for credit risk problems  

---

### 3. Global Model Performance

Evaluated on a held-out test set:

| Metric     | Value |
|-----------|-------|
| ROC-AUC   | ~0.71 |
| Precision | ~0.69 |
| Recall    | ~0.24 |
| F1-score  | ~0.36 |

The ROC curve shows **clear separation from random guessing**, indicating that the model learns meaningful risk patterns.

---

### 4. Group-Based Audit (Fairness Analysis)

Model performance was evaluated **separately for each age group**:

- 21–29  
- 30–39  
- 40–49  
- 50+  

#### Metrics analyzed per group:
- Default rate  
- ROC-AUC  
- Precision  
- Recall  

#### Key findings:
- Higher recall for older customers (50+)  
- Lower recall for younger groups (21–39), meaning more defaults go undetected  
- Ordering power (ROC-AUC) decreases for older age groups  

These results show that **model behavior is not uniform across groups**, which is critical in risk-sensitive applications.

---

### 5. Robustness Stress Test

To assess model stability, **Gaussian noise (5% of feature standard deviation)** was added to all numerical features in the test set.

#### Results:

- **ROC-AUC (original):** 0.7078  
- **ROC-AUC (stressed):** 0.7073  
- **Performance drop:** 0.0005  

#### Interpretation:
- The negligible performance drop indicates **high robustness**
- The model does not rely on fragile or overly precise feature values
- This is a strong signal of **good generalization quality**

---

## Key Takeaways

- The model performs reasonably well as a baseline classifier  
- There is measurable disparity in performance across age groups  
- The model is stable under moderate data perturbations  
- Group-level auditing reveals risks invisible when using global metrics alone  

---

## Why This Project Matters

This project goes beyond traditional machine learning tutorials by addressing:

- Group-level performance auditing  
- Bias and disparity analysis  
- Model robustness and reliability  
- Risk-aware evaluation practices  

These are **core skills expected from Data Scientists working in production environments**, especially in finance, credit risk, fraud detection, and regulated domains.

---

## Technologies Used

- Python  
- pandas, numpy  
- scikit-learn  
- matplotlib  

---

## Next Steps

Possible extensions include:
- Threshold optimization per group  
- Model calibration analysis  
- Alternative models (tree-based or boosting)  
- Drift monitoring simulations  
- Deployment-oriented reporting  

---
