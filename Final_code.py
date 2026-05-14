# ============================================================
# PROJECT: Predicting Student Academic Performance
# Using Regression Models
#
# Team Sections:
# - Mohit: Data cleaning, preprocessing, and visuals
# - Aman: Model development and training
# - Yuvika: Cross-validation and model evaluation visuals
# ============================================================


# ============================================================
# MOHIT SECTION: DATA CLEANING AND VISUALIZATION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the original dataset
df = pd.read_csv("StudentPerformanceFactors (1).csv")

# Clean column names:
# - Remove extra spaces
# - Convert names to lowercase
# - Replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Display column names to confirm cleaning
print("Columns in dataset:")
print(df.columns)

# Check missing values before cleaning
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Convert categorical variables into numeric dummy variables
df = pd.get_dummies(df, drop_first=True)

# Save the cleaned dataset
df.to_csv("cleaned_data.csv", index=False)

print("\nCleaned data saved as cleaned_data.csv")


# ======================
# VISUALS
# ======================

# Correlation heatmap to show relationships between variables
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation.png", bbox_inches="tight")
plt.close()

# Scatter plot: Hours Studied vs Exam Score
if "hours_studied" in df.columns and "exam_score" in df.columns:
    plt.figure(figsize=(7, 5))
    plt.scatter(df["hours_studied"], df["exam_score"])
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Hours Studied vs Exam Score")
    plt.savefig("study_vs_score.png", bbox_inches="tight")
    plt.close()

print("Visuals created and saved in the main project folder.")


# ============================================================
# AMAN SECTION: MODEL DEVELOPMENT AND TRAINING
# ============================================================

import os
import zipfile
import glob
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\nLibraries imported successfully.")


# ------------------------------------------------------------
# Load cleaned dataset
# ------------------------------------------------------------

csv_path = None

# First priority: use cleaned_data.csv if available
if os.path.exists("cleaned_data.csv"):
    csv_path = "cleaned_data.csv"
    print("Using cleaned_data.csv")

# Second priority: use any CSV file in the current folder
elif len(glob.glob("*.csv")) > 0:
    csv_path = glob.glob("*.csv")[0]
    print(f"Using CSV file found: {csv_path}")

# Third priority: extract CSV from a ZIP file if uploaded
else:
    zip_files = glob.glob("*.zip")

    if len(zip_files) == 0:
        raise FileNotFoundError(
            "No CSV or ZIP file found. Please upload cleaned_data.csv or the dataset ZIP file."
        )

    zip_path = zip_files[0]
    print(f"Extracting from ZIP file: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("dataset")

    csv_files = glob.glob("dataset/*.csv")
    csv_path = csv_files[0]
    print(f"Using extracted CSV file: {csv_path}")


# Read the selected CSV file
df = pd.read_csv(csv_path)

# Clean column names again to ensure consistency
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nDataset loaded successfully.")
print("Original shape:", df.shape)

# Display first few rows
print(df.head())


# ------------------------------------------------------------
# Prepare data for modeling
# ------------------------------------------------------------

# Target variable for prediction
target_col = "exam_score"

# Remove missing values
df_model = df.dropna().copy()

# Convert categorical variables into dummy variables
df_model = pd.get_dummies(df_model, drop_first=True)

# Convert Boolean columns to 0/1 integers
bool_cols = df_model.select_dtypes(include="bool").columns
df_model[bool_cols] = df_model[bool_cols].astype(int)

print("\nData prepared for modeling.")
print("Original shape:", df.shape)
print("After cleaning/encoding:", df_model.shape)

# Display first few rows of final modeling dataset
print(df_model.head())


# ------------------------------------------------------------
# Split features and target
# ------------------------------------------------------------

X = df_model.drop(columns=[target_col])
y = df_model[target_col]

print("\nFeature matrix shape:", X.shape)
print("Target variable shape:", y.shape)

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)


# ------------------------------------------------------------
# Define regression models
# ------------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree Regressor": DecisionTreeRegressor(
        random_state=42,
        max_depth=8
    ),

    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}


# ------------------------------------------------------------
# Train models
# ------------------------------------------------------------

trained_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[model_name] = model
    print(f"{model_name} trained successfully.")

print("All models trained successfully.")


# ------------------------------------------------------------
# Evaluate models using MAE, RMSE, and R²
# ------------------------------------------------------------

results = []

for model_name, model in trained_models.items():
    # Generate predictions on test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Store results
    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

# Convert results to DataFrame and sort by RMSE
results_df = pd.DataFrame(results).sort_values(by="RMSE")

print("\nModel Performance Results:")
print(results_df)


# ------------------------------------------------------------
# Save trained models and feature columns
# ------------------------------------------------------------

joblib.dump(trained_models, "aman_trained_regression_models.pkl")
joblib.dump(list(X.columns), "aman_model_feature_columns.pkl")

print("\nSaved files:")
print("- aman_trained_regression_models.pkl")
print("- aman_model_feature_columns.pkl")


# ------------------------------------------------------------
# Aman Summary
# ------------------------------------------------------------

"""
Aman: Model Development and Training Summary

For my part of the project, I completed the model development and training stage.
I used exam_score as the target variable and prepared the cleaned dataset for
regression modeling. I split the data into training and testing sets using an
80/20 split, then trained three regression models: Linear Regression, Decision
Tree Regressor, and Random Forest Regressor.

After training, I generated predictions on the test set and compared model
performance using MAE, RMSE, and R² score. I also saved the trained models and
feature column list so they can be used later for evaluation, documentation, or
the final project repository.
"""


# ============================================================
# YUVIKA SECTION: CROSS-VALIDATION AND MODEL COMPARISON
# ============================================================

from sklearn.model_selection import cross_val_score

print("\nCross-Validation R² Scores:\n")

cv_results = []

for model_name, model in trained_models.items():
    # 5-fold cross-validation using R² score
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    cv_results.append({
        "Model": model_name,
        "Mean CV R2": scores.mean(),
        "Standard Deviation": scores.std()
    })

    print(f"{model_name}: {scores.mean():.3f}")

cv_results_df = pd.DataFrame(cv_results)

print("\nCross-Validation Results:")
print(cv_results_df)


# ------------------------------------------------------------
# Chart 1: Model comparison using R² score
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["R2 Score"])
plt.title("Model Comparison (R²)")
plt.ylabel("R² Score")
plt.xticks(rotation=20)
plt.savefig("r2_comparison.png", bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# Chart 2: Cross-validation performance
# Added because project requirements mention cross-validation.
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(cv_results_df["Model"], cv_results_df["Mean CV R2"])
plt.title("Cross-Validation Performance")
plt.ylabel("Mean CV R² Score")
plt.xticks(rotation=20)
plt.savefig("cross_validation_performance.png", bbox_inches="tight")
plt.show()


# ============================================================
# ADDITIONAL REQUIRED SECTION:
# MODEL PERFORMANCE ON DIFFERENT TRAINING SIZES
# ============================================================

# This section is added because the project requirements specifically ask
# for model performance on different training sizes.

training_sizes = [0.40, 0.60, 0.80, 1.00]
training_size_results = []

# Using Linear Regression here because the report says it performed best.
for size in training_sizes:

    # For 40%, 60%, and 80%, use a subset of the training data
    if size < 1.00:
        X_subset, _, y_subset, _ = train_test_split(
            X_train,
            y_train,
            train_size=size,
            random_state=42
        )

    # For 100%, use the full training data
    else:
        X_subset = X_train
        y_subset = y_train

    # Train Linear Regression model on the selected training size
    model = LinearRegression()
    model.fit(X_subset, y_subset)

    # Predict on the same fixed test set
    y_pred = model.predict(X_test)

    # Store performance metrics
    training_size_results.append({
        "Training Size": size,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    })

training_size_df = pd.DataFrame(training_size_results)

print("\nModel Performance on Different Training Sizes:")
print(training_size_df)


# ------------------------------------------------------------
# Chart 3: Training-size performance
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(
    training_size_df["Training Size"],
    training_size_df["R2 Score"],
    marker="o"
)
plt.title("Model Performance on Different Training Sizes")
plt.xlabel("Training Size")
plt.ylabel("R² Score")
plt.savefig("training_size_performance.png", bbox_inches="tight")
plt.show()


# ============================================================
# FEATURE IMPORTANCE CHART
# ============================================================

# Use Random Forest feature importance because this matches

rf_model = trained_models["Random Forest Regressor"]

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

top_10_features = feature_importance_df.head(10)

print("\nTop 10 Important Features:")
print(top_10_features)

plt.figure(figsize=(9, 6))
plt.barh(top_10_features["Feature"], top_10_features["Importance"])
plt.xlabel("Importance")
plt.title("Top 10 Important Features Affecting Exam Scores")
plt.gca().invert_yaxis()
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.show()

