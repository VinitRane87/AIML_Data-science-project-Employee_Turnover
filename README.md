# Employee Turnover Analytics - Portobello Tech

**Applied Data Science with Python - Course-End Project 1**

## Problem Statement

Portobello Tech wants to predict employee turnover using historical work data: satisfaction, evaluation scores, project count, hours worked, tenure, promotions, and salary level. As the ML Developer assigned to the HR Department, this project builds and evaluates models to predict which employees are at risk of leaving, and translates those predictions into concrete retention strategies.

## Data

`HR_comma_sep.csv` - 14,999 employee records (source: Kaggle HR dataset) with satisfaction level, last evaluation score, number of projects, average monthly hours, tenure, work accident history, promotions in the last 5 years, department, salary band, and the target variable `left` (1 = left the company).

## Approach

The notebook (`Employee_Turnover_Analytics.ipynb`) follows the seven required steps:

1. Data quality checks - confirmed no missing values; flagged duplicate rows for follow-up with HR.
2. EDA - correlation heatmap, distribution plots, and a project-count bar chart showing a U-shaped attrition pattern.
3. K-Means clustering (k=3) of employees who left, based on satisfaction and evaluation.
4. SMOTE - encoded categorical variables, stratified 80:20 split (random_state=123), upsampled the training set only.
5. 5-fold cross-validation on Logistic Regression, Random Forest, and Gradient Boosting.
6. Best model selection using ROC/AUC and confusion matrices; explained why Recall matters more than Precision here.
7. Retention strategy - risk zones (Safe/Low/Medium/High) with targeted recommendations per zone.

## Key Findings

- Random Forest was the best-performing model (AUC approximately 0.995).
- Dissatisfaction is the strongest driver of turnover.
- Both under- and over-utilization (project count) drive attrition.

## Files

| File | Description |
|---|---|
| `Employee_Turnover_Analytics.ipynb` | Full Jupyter notebook |
| `Employee_Turnover_Analytics.html` | HTML export |
| `Employee_Turnover_Analytics.pdf` | PDF export |
| `HR_comma_sep.csv` | Source dataset |

## Tools

Python, Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE), Matplotlib, Seaborn, JupyterLab.****
