# Home-Credit-Default-Risk

📊 Home Credit Default Risk — Project Summary
Overview

This project focuses on predicting the probability of loan default using data from the Home Credit Default Risk competition on Kaggle. The dataset includes multiple tables and hundreds of engineered and behavioral features describing clients, previous applications, credit bureau records, and payment histories.

The primary goal was to build a robust machine learning pipeline that effectively handles the high feature dimensionality and delivers strong predictive performance measured by AUC (Area Under the ROC Curve).

🔍 Exploratory Data Analysis (EDA)
1. Understanding the Data Structure

The raw dataset contains over 600 features spread across multiple relational tables. The initial step involved:

Exploring feature types (numerical, categorical, time-based).

Identifying key groups: client characteristics, previous loans, credit bureau records, installment history, and POS cash data.

Evaluating missing value patterns and sparsity across tables.

2. Data Quality Assessment

Several datasets contained extensive missing values and outliers. I performed:

Missing value heatmaps and distribution checks.

Outlier detection for financial variables (e.g., credit amounts, annuities, income).

Consistency checks between related fields (e.g., credit duration vs. payment history).

3. Feature Understanding & Target Relationships

To understand which attributes were most predictive of default risk, I examined:

Target distributions and class balance.

Correlations between financial stress indicators and default outcomes.

Univariate and bivariate visualizations such as:

Income vs. credit amount ratios

Payment behavior curves

Days before application (behavioral time markers)

This stage guided the feature engineering and selection process by highlighting which types of variables carried most predictive signal.

🧮 Feature Selection (600+ → ~90 Features)

With hundreds of raw and engineered features, reducing dimensionality was essential for model interpretability and generalization. The final selection combined statistical, model-based, and domain-driven methods:

1. Removing Low-Information Features

Dropped features with extreme sparsity.

Removed features with no variance or very low variance.

Pruned redundant features with high mutual correlation.

2. Model-Based Importance Analysis

Utilized feature importance scores from:

LightGBM

XGBoost

By iteratively training baseline models and removing low-impact features, I ensured that only variables contributing meaningful predictive power were retained.

3. Domain-Informed Filtering

Some features, although statistically significant, were inconsistent or overly noisy. Based on EDA insights:

Consolidated similar engineered metrics (e.g., various ratio-based features).

Prioritized behavioral and historical payment features that consistently showed strong predictive value.

This multi-step process reduced the feature set from 600+ to around 90 high-quality features, improving model efficiency without sacrificing predictive capacity.

🚀 Model Development & Performance Boosting
1. Baseline Model

The starting point was a gradient boosting model achieving an AUC of ~0.74, similar to typical first-pass benchmarks for this competition.

2. Advanced Modeling Enhancements

To improve predictive performance, I incorporated:

Tuned LightGBM/XGBoost hyperparameters (learning rate, depth, regularization).

Early stopping and cross-validation for stability.

Handling class imbalance via balanced weights.

Feature interaction awareness through boosted tree modeling.

3. Final Model Performance

By combining effective feature selection with optimized boosting models, I increased the model's performance from:

🔹 0.74 AUC → 🔹 ~0.79 AUC

This improvement demonstrates:

Cleaner, more expressive features

Reduced overfitting

Stronger learning of non-linear patterns in customer behavioral data

📌 Key Takeaways

Conducted deep EDA across multiple complex relational datasets.

Reduced feature dimensionality by ~85% while preserving predictive signal.

Improved model performance by ~5 AUC points, reaching competitive levels.

Built a robust pipeline using gradient boosting techniques optimized for tabular data.
