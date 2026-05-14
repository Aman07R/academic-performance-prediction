# Student Performance Regression

This project predicts student exam scores using regression models and academic performance factors such as study habits, attendance, previous scores, lifestyle habits, and support resources.

## Overview

The goal of this project is to compare different regression models and identify which one best predicts student exam performance.

The target variable is:

```text
exam_score
```

## Dataset

The dataset contains student academic performance records with features related to:

• Study habits

• Attendance

• Previous academic performance

• Sleep and physical activity

• Tutoring and academic support

• Family income and learning resources

### Dataset summary:

| Item                    |      Value |
| ----------------------- | ---------: |
| Original records        |      6,607 |
| Cleaned records         |      6,378 |
| Original columns        |         20 |
| Final modeling features |         27 |
| Target variable         | exam_score |


## Models Used

The project compares three regression models:

• Linear Regression

• Decision Tree Regressor

• Random Forest Regressor

The models were evaluated using:

• Mean Absolute Error

• Root Mean Squared Error

• R² Score

• Cross-validation R² Score

## Results

| Model                   |   MAE |  RMSE | R² Score |
| ----------------------- | ----: | ----: | -------: |
| Linear Regression       | 0.487 | 2.043 |    0.731 |
| Random Forest Regressor | 1.237 | 2.440 |    0.617 |
| Decision Tree Regressor | 1.674 | 3.136 |    0.367 |


Linear Regression performed best overall, with the highest R² score and the lowest RMSE.

## Key Findings
• Attendance was one of the strongest predictors of exam scores.

• Hours studied had a positive relationship with academic performance.

• Previous scores were also highly useful for prediction.

• Lifestyle factors had a smaller effect compared to academic factors.


## Files
| File                                | Description                     |
| ----------------------------------- | ------------------------------- |
| `Final_code.py`                     | Main project code               |
| `StudentPerformanceFactors (1).csv` | Original dataset                |
| `cleaned_data.csv`                  | Cleaned dataset                 |
| `model_results.csv`                 | Model performance results       |
| `correlation.png`                   | Correlation heatmap             |
| `study_vs_score.png`                | Study hours vs exam score chart |
| `r2_comparison.png`                 | Model comparison chart          |
| `cross_validation_performance.png`  | Cross-validation chart          |
| `training_size_performance.png`     | Training-size performance chart |
| `feature_importance.png`            | Feature importance chart        |



## Tools Used
• Python

• Pandas

• NumPy

• Matplotlib

• Seaborn

• Scikit-learn

• Joblib

## Conclusion
This project shows that simple regression models can effectively predict student exam performance when academic behavior and support-related factors are included. Linear Regression performed best, suggesting that the relationship between the selected features and exam scores is mostly linear.
