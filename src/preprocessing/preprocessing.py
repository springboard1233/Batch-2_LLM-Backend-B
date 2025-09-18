# Step 1: Data Collection & Inspection
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"F:\Python\Batch-2_LLM-Backend-B\data\processed\bfsi_cleaned_transactions.csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nColumns in dataset:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:\n", df.describe())

# Step 2: Exploratory Data Analysis (EDA)
sns.set(style="whitegrid")   # Visualization style

# 1️⃣ Fraud distribution
plt.figure(figsize=(6,4))
sns.countplot(x="is_fraud", hue="is_fraud", data=df, palette="coolwarm", legend=False)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

# 2️⃣ Fraud distribution across channels
plt.figure(figsize=(8,5))
sns.countplot(x="channel", hue="is_fraud", data=df, palette="coolwarm")
plt.title("Fraud Distribution Across Channels")
plt.show()

# 3️⃣ Fraud distribution by KYC verification
plt.figure(figsize=(6,4))
sns.countplot(x="kyc_verified", hue="is_fraud", data=df, palette="coolwarm")
plt.title("Fraud by KYC Verification Status")
plt.show()

# 4️⃣ Transaction amount distribution (fraud vs non-fraud)
plt.figure(figsize=(8,5))
sns.boxplot(x="is_fraud", y="transaction_amount", hue="is_fraud",
            data=df, palette="coolwarm", legend=False)
plt.title("Transaction Amount Distribution by Fraud Status")
plt.ylim(0, 200000)  # Limit for better view
plt.show()

# 5️⃣ Account age distribution (fraud vs non-fraud)
plt.figure(figsize=(8,5))
sns.boxplot(x="is_fraud", y="account_age_days", hue="is_fraud",
            data=df, palette="coolwarm", legend=False)
plt.title("Account Age Distribution by Fraud Status")
plt.show()

# 6️⃣ Fraud distribution by hour of the day
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

plt.figure(figsize=(10,5))
sns.countplot(x="hour", hue="is_fraud", data=df, palette="coolwarm")
plt.title("Fraud Distribution by Hour of Day")
plt.show()

# Step 3: Data Cleaning

# 1. Check for missing values
print("Missing Values:\n", df.isnull().sum())

# 2. Check for duplicates
duplicates = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicates)

# Drop duplicates if any
df = df.drop_duplicates()

# 3. Convert categorical variables to category dtype
categorical_cols = ["kyc_verified", "channel"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

# 4. Check for outliers using IQR method for transaction_amount
Q1 = df['transaction_amount'].quantile(0.25)
Q3 = df['transaction_amount'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[(df['transaction_amount'] < lower_limit) | (df['transaction_amount'] > upper_limit)]
print("\nNumber of outliers in transaction_amount:", len(outliers))

# Option: Cap outliers (winsorization)
df['transaction_amount'] = df['transaction_amount'].clip(lower_limit, upper_limit)

# 5. Re-check dataset info after cleaning
print("\nDataset Info After Cleaning:")
print(df.info())
# Step 4: Feature Engineering

import numpy as np

# Ensure timestamp is in datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 1. Extract datetime features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon, 6=Sun
df["month"] = df["timestamp"].dt.month

# 2. Log-transform transaction amount (to reduce skewness)
df["log_amount"] = np.log1p(df["transaction_amount"])

# 3. Customer-level aggregated features
customer_stats = df.groupby("customer_id")["transaction_amount"].agg(
    ["count", "mean", "max", "min"]
).reset_index()

customer_stats.rename(columns={
    "count": "cust_txn_count",
    "mean": "cust_avg_amount",
    "max": "cust_max_amount",
    "min": "cust_min_amount"
}, inplace=True)

# Merge back into main df
df = df.merge(customer_stats, on="customer_id", how="left")

# 4. Drop unused columns (transaction_id, timestamp, customer_id) for ML
df_ml = df.drop(["transaction_id", "timestamp", "customer_id"], axis=1)

print("Feature Engineered Dataset Shape:", df_ml.shape)
print("\nColumns after feature engineering:\n", df_ml.columns.tolist())

# Step 5: Feature Selection (fixed version)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Split features and target
X = df_ml.drop("is_fraud", axis=1)
y = df_ml["is_fraud"].astype(int)

# --- 1️⃣ Correlation Heatmap (numeric features only) ---
numeric_X = X.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(12,8))
sns.heatmap(numeric_X.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# --- 2️⃣ Chi-Square Test (categorical vs target) ---
cat_features = ["kyc_verified", "channel"]
X_cat = pd.get_dummies(df_ml[cat_features], drop_first=True)

chi2_selector = SelectKBest(score_func=chi2, k="all")
chi2_selector.fit(X_cat, y)

chi2_scores = (
    pd.DataFrame({
        "Feature": X_cat.columns,
        "Chi2 Score": chi2_selector.scores_
    })
    .sort_values(by="Chi2 Score", ascending=False)
)
print("\nChi-Square Test Results:\n", chi2_scores)

# --- 3️⃣ Feature Importance using RandomForest ---
X_encoded = pd.get_dummies(X, drop_first=True)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_encoded, y)

importances = (
    pd.DataFrame({
        "Feature": X_encoded.columns,
        "Importance": rf_model.feature_importances_
    })
    .sort_values(by="Importance", ascending=False)
)
print("\nTop Features by RandomForest:\n", importances.head(10))


plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature",
            data=importances.head(10),
            color="steelblue")   # use color instead of palette
plt.title("Top 10 Important Features (RandomForest)")
plt.show()

# Step 6: Train-Test Split

from sklearn.model_selection import train_test_split

# Convert categorical variables into numeric (One-Hot Encoding)
X_encoded = pd.get_dummies(X, drop_first=True)

# Target variable
y = df_ml["is_fraud"].astype(int)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Check target distribution in both sets
print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))

# Step 7: Model Selection & Training

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

# Store results
results = []

# Train & evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append([name, acc, prec, rec, f1])

# Convert to DataFrame for comparison
import pandas as pd
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])

print("Model Comparison Results:\n")
print(results_df)

# Highlight best model
best_model = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]
print("\nBest Model:", best_model["Model"])


# Step 8: Save Processed Data & Model

import os
import joblib

# Create folder if it doesn't exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Save processed dataset
df_ml.to_csv(r"F:\Python\Batch-2_LLM-Backend-B\data\processed\transactions_processed.csv", index=False)
print("✅ Processed dataset saved at data/processed/transactions_processed.csv")

# 2. Save the best model
best_model_name = best_model["Model"]
final_model = models[best_model_name]
joblib.dump(final_model, f"models/{best_model_name}_fraud_model.pkl")

print(f"✅ Best model ({best_model_name}) saved at models/{best_model_name}_fraud_model.pkl")
