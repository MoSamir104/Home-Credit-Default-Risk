# Home Credit Default Risk

### Machine Learning Model for Predicting Loan Default Using Advanced Feature Engineering and Gradient Boosting

---

## Overview

This project addresses the Home Credit Default Risk competition on Kaggle, where the goal is to build a model that predicts the probability of a loan applicant defaulting on a loan. The dataset consists of multiple relational tables and more than 600 raw and engineered features describing demographic information, income, employment, previous loans, payment behavior, and credit bureau history.

The objective of the project was to:

1. Perform comprehensive Exploratory Data Analysis (EDA) across all available tables.
2. Reduce the feature space from over 600 features to a compact and highly predictive set of approximately 90 features.
3. Build and optimize a gradient boosting model that improves baseline performance.
4. Achieve strong predictive accuracy measured using the Area Under the ROC Curve (AUC).

The final model improved baseline performance from approximately 0.74 AUC to about 0.79 AUC after feature selection and model optimization.

---

## Table of Contents

* Overview
* Data Description
* Exploratory Data Analysis
* Feature Engineering and Selection
* Selected Features Explanation
* Modeling Approach
* Results
* Key Learnings
* Tools and Technologies

---

## Data Description

The Home Credit Default Risk dataset includes multiple relational tables:

* **application_train / application_test**: Main client-level features
* **bureau / bureau_balance**: Credit bureau history and monthly balances
* **previous_application**: Data on previous loan applications
* **POS_CASH_balance**: Information on POS and cash loan transactions
* **installments_payments**: Installment payment histories
* **credit_card_balance**: Credit card monthly balances

Features include demographic details, income, employment duration, payment performance, behavioral patterns, financial ratios, and engineered metrics.

---

## Exploratory Data Analysis (EDA)

### Data Structure Understanding

The dataset spans several relational sources. During EDA, the focus was on:

* Understanding data relationships across tables
* Identifying numerical, categorical, and time-based variables
* Mapping missing values and sparsity
* Detecting inconsistencies and outliers

### Data Cleaning and Quality Checks

To ensure model robustness, the following steps were completed:

* Removal or imputation of missing values in key variables
* Handling of outliers in financial fields
* Distribution analysis and normalization where required
* Validation of time-based and sequential features

### Target Relationship Analysis

Feature-target relationships were analyzed to identify key risk indicators, such as:

* Payment delays
* Credit-to-income ratios
* Credit duration
* Behavioral trends in past applications

This analysis guided the feature engineering and selection process.

---

## Feature Engineering and Selection

The original dataset contained more than 1000 features. Dimensionality reduction was necessary to improve model interpretability, reduce noise, and enhance performance.

### Step 1: Initial Feature Pruning

* Removed constant and near-constant features
* Dropped features with extreme sparsity
* Eliminated redundant or duplicated variables

### Step 2: Correlation-Based Filtering

* Removed highly correlated features to minimize multicollinearity
* Consolidated engineered ratios with similar signal

### Step 3: Model-Based Selection

Using LightGBM and XGBoost:

* Feature importance scores were analyzed iteratively
* Low-impact features were removed in multiple pruning cycles
* Features demonstrating strong predictive signal were retained

### Final Feature Set

After the complete selection process, the number of features was reduced to approximately **90**. The reduced feature set improved model performance and significantly lowered overfitting.

---

## Selected Features Explanation

The final selected features fall into the following major categories:

### 1. External Risk and Scoring Features

These represent aggregated external risk evaluations and are among the most predictive features.

Examples:
EXT_MEAN, EXT_MIN, EXT_MAX, EXT_STD, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3

### 2. Financial Ratios and Behavioral Metrics

These features normalize financial information and highlight risk-related patterns.

Examples:
ANNUITY_TO_CREDIT, ANNUITY_TO_PRICE, DEBT_RATIO, CREDIT_TO_PRICE, LIMIT_USAGE, CAR_AGE_TO_AGE

### 3. Loan and Financial Amount Features

Core financial variables capturing credit size, payment amounts, and overdue behavior.

Examples:
AMT_CREDIT, AMT_GOODS_PRICE, AMT_CREDIT_SUM, AMT_CREDIT_MAX_OVERDUE, AMT_PAYMENT_MIN, AMT_PAYMENT_MEAN, AMT_INSTALMENT_MAX, AMT_INSTALMENT_STD

### 4. Time-Based Employment and Identity Features

These capture lifecycle and behavioral stability.

Examples:
YEARS_EMPLOYED, YEARS_ID_PUBLISH, Years_Credit, AGE, DAYS_CREDIT_ENDDATE, DAYS_INSTALMENT_MIN/MAX/STD/SUM

### 5. Previous Application History

Historical loan performance and behavior indicators.

Examples:
PREV_APPROVED_RATIO, PREV_REFUSED_RATIO, PREV_CNT_PAYMENT_MEAN, PREV_AMT_DOWN_PAYMENT_MAX, PREV_AMT_ANNUITY_MEAN

### 6. POS Cash Loan and Installment Features

Indicators of delinquency and future obligations.

Examples:
POS_CNT_INSTALMENT_FUTURE_MEAN, POS_CNT_INSTALMENT_MAX, POS_SK_DPD_DEF_MEAN

### 7. Demographic and Social Features

Background characteristics relating to risk segmentation.

Examples:
OCCUPATION_TYPE, ORGANIZATION_TYPE, NAME_EDUCATION_TYPE, REGION_RATING_CLIENT_W_CITY, CODE_GENDER

### 8. Product Combination Features

Summaries of product types used in past applications.

Examples:
PRODUCT_COMBINATION_Cash_Street_high_MEAN, PRODUCT_COMBINATION_Cash_X_Sell_high_MEAN, PRODUCT_COMBINATION_Cash_X_Sell_low_MEAN

### 9. Variability and Volatility Features

Indicators of instability in payments or installment schedules.

Examples:
NUM_INSTALMENT_NUMBER_STD, NUM_INSTALMENT_VERSION_STD, DAYS_ENTRY_PAYMENT_STD

### 10. Missingness Indicators

Missing value flags that carry predictive information.

Examples:
OWN_CAR_AGE_MISSING, DEBT_RATIO_MISSING, POS_CNT_INSTALMENT_MAX_MISSING, PREV_CNT_PAYMENT_MEAN_MISSING, AMT_PAYMENT_MEAN_MISSING

### 11. Core Identifiers

SK_ID_CURR (unique identifier)
TARGET (default indicator)

---

## Modeling Approach

### Baseline Model

A first-pass gradient boosting model achieved around 0.74 AUC.

### Model Enhancements

To improve the model:

* Hyperparameters were optimized (depth, learning rate, regularization)
* K-fold cross-validation ensured stability
* Early stopping prevented overfitting
* Class imbalance was addressed with weighting
* Feature interactions were captured through boosted trees

### Final Model

The optimized LightGBM/XGBoost model gained significantly from the cleaned feature set and parameter tuning.

---

## Results

### Performance

* **Baseline AUC:** ~0.74
* **Final AUC after feature selection and tuning:** ~0.79

### Additional Improvements

* Reduced training time
* Lower variance across folds
* Higher model stability
* Improved interpretability through a smaller feature set

---

## Key Learnings

* High-dimensional tabular datasets require careful feature cleaning and reduction.
* External source scores and behavioral features are highly predictive.
* Missingness itself contains important information in financial risk data.
* Gradient boosting models perform best on structured, well-engineered feature sets.
* Iterative feature pruning significantly improves generalization.

---
