# ============================================================
# Predicting Student Academic Performance Using Regression Models
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_PATH = "StudentPerformanceFactors (1).csv"
TARGET_COL = "exam_score"
RANDOM_STATE = 42


# ------------------------------------------------------------
# Load and clean data
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)

# Remove missing values
df = df.dropna()

# Encode categorical variables
df_model = pd.get_dummies(df, drop_first=True)

# Convert boolean columns to integers
bool_cols = df_model.select_dtypes(include="bool").columns
df_model[bool_cols] = df_model[bool_cols].astype(int)

# Save cleaned dataset
df_model.to_csv("cleaned_data.csv", index=False)

print("Cleaned dataset saved as cleaned_data.csv")
print("Final modeling shape:", df_model.shape)


# ------------------------------------------------------------
# Exploratory visualizations
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
sns.heatmap(df_model.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation.png", bbox_inches="tight")
plt.close()

if "hours_studied" in df_model.columns and TARGET_COL in df_model.columns:
    plt.figure(figsize=(7, 5))
    plt.scatter(df_model["hours_studied"], df_model[TARGET_COL])
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Hours Studied vs Exam Score")
    plt.savefig("study_vs_score.png", bbox_inches="tight")
    plt.close()

print("Exploratory visualizations saved.")


# ------------------------------------------------------------
# Prepare features and target
# ------------------------------------------------------------

X = df_model.drop(columns=[TARGET_COL])
y = df_model[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)


# ------------------------------------------------------------
# Define and train models
# ------------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(
        random_state=RANDOM_STATE,
        max_depth=8
    ),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

trained_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[model_name] = model
    print(f"{model_name} trained successfully.")


# ------------------------------------------------------------
# Evaluate models
# ------------------------------------------------------------

results = []

for model_name, model in trained_models.items():
    y_pred = model.predict(X_test)

    results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values(by="RMSE")
results_df.to_csv("model_results.csv", index=False)

print("\nModel Performance Results:")
print(results_df)


# ------------------------------------------------------------
# Cross-validation
# ------------------------------------------------------------

cv_results = []

for model_name, model in trained_models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    cv_results.append({
        "Model": model_name,
        "Mean CV R2": scores.mean(),
        "CV Standard Deviation": scores.std()
    })

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv("cross_validation_results.csv", index=False)

print("\nCross-Validation Results:")
print(cv_results_df)


# ------------------------------------------------------------
# Model comparison chart
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["R2 Score"])
plt.title("Model Comparison by R² Score")
plt.ylabel("R² Score")
plt.xticks(rotation=20)
plt.savefig("r2_comparison.png", bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# Cross-validation chart
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(cv_results_df["Model"], cv_results_df["Mean CV R2"])
plt.title("Cross-Validation Performance")
plt.ylabel("Mean CV R² Score")
plt.xticks(rotation=20)
plt.savefig("cross_validation_performance.png", bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# Training-size performance
# ------------------------------------------------------------

training_sizes = [0.40, 0.60, 0.80, 1.00]
training_size_results = []

best_model = LinearRegression()

for size in training_sizes:
    if size < 1.00:
        X_subset, _, y_subset, _ = train_test_split(
            X_train,
            y_train,
            train_size=size,
            random_state=RANDOM_STATE
        )
    else:
        X_subset = X_train
        y_subset = y_train

    best_model.fit(X_subset, y_subset)
    y_pred = best_model.predict(X_test)

    training_size_results.append({
        "Training Size": size,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    })

training_size_df = pd.DataFrame(training_size_results)
training_size_df.to_csv("training_size_results.csv", index=False)

print("\nTraining Size Performance:")
print(training_size_df)

plt.figure(figsize=(8, 5))
plt.plot(
    training_size_df["Training Size"],
    training_size_df["R2 Score"],
    marker="o"
)
plt.title("Model Performance Across Training Sizes")
plt.xlabel("Training Size")
plt.ylabel("R² Score")
plt.savefig("training_size_performance.png", bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

rf_model = trained_models["Random Forest Regressor"]

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

top_10_features = feature_importance_df.head(10)
feature_importance_df.to_csv("feature_importance.csv", index=False)

print("\nTop 10 Important Features:")
print(top_10_features)

plt.figure(figsize=(9, 6))
plt.barh(top_10_features["Feature"], top_10_features["Importance"])
plt.xlabel("Importance")
plt.title("Top 10 Features Affecting Exam Scores")
plt.gca().invert_yaxis()
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# Save trained models and feature columns
# ------------------------------------------------------------

joblib.dump(trained_models, "trained_regression_models.pkl")
joblib.dump(list(X.columns), "model_feature_columns.pkl")

print("\nSaved project outputs:")
print("- cleaned_data.csv")
print("- model_results.csv")
print("- cross_validation_results.csv")
print("- training_size_results.csv")
print("- feature_importance.csv")
print("- trained_regression_models.pkl")
print("- model_feature_columns.pkl")
print("- visualization files")
