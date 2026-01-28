📊 Credit Risk Model Audit: Fairness, Robustness & Stability
Overview

This project presents an end-to-end audit of a credit risk classification model, focusing not only on predictive performance, but also on fairness across groups, model stability, and robustness to data perturbations.

Instead of optimizing for accuracy alone, the goal is to answer production-level questions such as:

Does the model perform consistently across different demographic groups?

Are some groups systematically under- or over-penalized?

Is the model stable when the input data slightly changes?

This reflects real-world responsibilities of a Data Scientist working with risk-sensitive models.

Problem Statement

Given historical credit card customer data, the task is to predict default risk (binary classification):

DEFAULT = 1 → customer failed to pay (default)

DEFAULT = 0 → customer paid normally

Beyond prediction, the project emphasizes:

group-level performance auditing

bias and disparity analysis

robustness under stress scenarios

Dataset

Source: UCI Credit Card Default Dataset

Size: ~30,000 records

Target: DEFAULT

Features include:

Credit limit

Demographics (age, sex, education, marital status)

Payment history

Billing and payment amounts

An additional feature, AGE_BIN, was created to enable age-based group auditing.

Methodology
1. Data Preparation

Removed duplicated headers and non-informative columns

Converted all numerical features to proper numeric types

Created categorical age groups (AGE_BIN)

Stratified train/test split to preserve default rate distribution

2. Modeling Pipeline

A fully reproducible pipeline was built using scikit-learn:

Numerical features

Median imputation

Standard scaling

Categorical features

Most-frequent imputation

One-hot encoding

Model

Logistic Regression (baseline, interpretable, industry-standard)

All preprocessing steps are included inside the pipeline to avoid data leakage.

3. Global Model Performance

Evaluated on a held-out test set:

Metric	Value
ROC-AUC	~0.71
Precision	~0.69
Recall	~0.24
F1-score	~0.36

The ROC curve shows clear separation from random guessing, indicating that the model learns meaningful risk patterns.

4. Group-Based Audit (Fairness Analysis)

Performance was evaluated separately for each age group:

21–29

30–39

40–49

50+

Metrics analyzed per group:

Default rate

ROC-AUC

Precision

Recall

Key Findings:

The model shows higher recall for older customers (50+)

Lower recall for younger groups (21–39), meaning more defaults go undetected

Ordering power (AUC) decreases for older groups

This demonstrates that model behavior is not uniform across groups, which is critical for risk-sensitive applications.

5. Robustness Stress Test

To assess model stability, Gaussian noise (5% of feature standard deviation) was added to all numerical features in the test set.

Scenario	ROC-AUC
Original data	0.7078
Stressed data	0.7073
Performance drop	0.0005
Interpretation:

The negligible performance drop indicates high robustness

The model does not rely on fragile or overly precise feature values

This is a strong signal of generalization quality

Key Takeaways

The model performs reasonably well as a baseline classifier

There is measurable disparity in performance across age groups

The model is stable under moderate data perturbations

Auditing reveals risks that would be invisible using global metrics alone

Why This Project Matters

This project goes beyond traditional ML tutorials by addressing:

Fairness and group-level performance

Model reliability and robustness

Risk-aware evaluation practices

These are core skills expected from Data Scientists working in production environments, especially in finance, credit, fraud, and regulated domains.

Technologies Used

Python

pandas, numpy

scikit-learn

matplotlib