import os, joblib, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# Optional advanced models & imbalance handling
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# -------------------- Feature Engineering --------------------
def feature_engineering(df):
    df = df.copy()
    
    # Drop IDs
    for col in ["transaction_id", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Timestamp features
    if "timestamp" in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].notna().any():
            df['hour'] = df['timestamp'].dt.hour
            df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df = df.drop(columns=['timestamp'])
    
    # Amount features
    if 'amount' in df.columns:
        df['log_amount'] = np.log1p(df['amount'])
        df['high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['high_amount_night'] = df['high_amount'] * df.get('is_night', 0)
    
    # KYC feature
    if "kyc_verified" in df.columns:
        df["kyc_verified"] = df["kyc_verified"].map({"Yes": 1, "No": 0}).fillna(0)
    
    # Transaction velocity: number of transactions per customer in last 24h
    if 'customer_id' in df.columns:
        df['customer_id_temp'] = df['customer_id']  # keep temporarily for velocity
        df['timestamp_temp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.sort_values(['customer_id_temp', 'timestamp_temp'], inplace=True)
        df['txn_last_24h'] = df.groupby('customer_id_temp')['timestamp_temp'].rolling('24h').count().reset_index(0, drop=True).fillna(0)
        df.drop(['customer_id_temp','timestamp_temp'], axis=1, inplace=True)
    
    # Categorical encoding
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [c for c in cat_cols if c != 'is_fraud']
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    df = df.fillna(0)
    return df

# -------------------- Model Definitions --------------------
def get_models():
    models = {
        "LogisticRegression": LogisticRegression(
            C=0.1, class_weight='balanced', max_iter=1000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=10, random_state=42,
            use_label_encoder=False, eval_metric='aucpr'
        )
    return models

# -------------------- Threshold Optimization --------------------
def find_optimal_threshold(y_true, y_probs, target_recall=0.80):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    valid_idx = np.where(recall[:-1] >= target_recall)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(precision[valid_idx])]
        return thresholds[best_idx], precision[best_idx], recall[best_idx]
    else:
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx], precision[best_idx], recall[best_idx]

# -------------------- Training Pipeline --------------------
def train_fraud_model(file_path, target_recall=0.8):
    df = pd.read_csv(file_path)
    df = feature_engineering(df)
    
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if HAS_SMOTE:
        sm = SMOTE(random_state=42)
        X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
    
    models = get_models()
    results = []
    best_model = None
    best_probs = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        ap = average_precision_score(y_test, probs)
        roc = roc_auc_score(y_test, probs)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        fraud_recall = report.get('1', {}).get('recall', 0)
        fraud_precision = report.get('1', {}).get('precision', 0)
        fraud_f1 = report.get('1', {}).get('f1-score', 0)
        
        results.append({
            "Model": name, "PR-AUC": ap, "ROC-AUC": roc,
            "Fraud_Recall": fraud_recall, "Fraud_Precision": fraud_precision, "Fraud_F1": fraud_f1
        })
        
        if ap > best_score:
            best_model = model
            best_score = ap
            best_probs = probs
            best_name = name
    
    optimal_thresh, opt_precision, opt_recall = find_optimal_threshold(y_test, best_probs, target_recall)
    final_preds = (best_probs >= optimal_thresh).astype(int)
    
    # Business impact metrics
    cm = confusion_matrix(y_test, final_preds)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle cases where only one class is present
        tn = cm[0,0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0,1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1,0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1,1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    fraud_caught_pct = tp / (tp + fn + 1e-8) * 100  # avoid divide by zero
    false_alarm_pct = fp / (fp + tn + 1e-8) * 100   
    print(f"\nBEST MODEL: {best_name} (PR-AUC: {best_score:.4f})")
    print(f"Optimal Threshold: {optimal_thresh:.4f}")
    print(f"Achieved Recall: {opt_recall:.4f}, Precision: {opt_precision:.4f}")
    print(f"Fraud Caught: {fraud_caught_pct:.2f}%, False Alarms: {false_alarm_pct:.2f}%")
    print(classification_report(y_test, final_preds, digits=4))
    
    output_dir = r"F:\Python\Batch-2_LLM-Backend-B\fraud_model_final"
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(best_model, f"{output_dir}/model.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(X.columns.tolist(), f"{output_dir}/features.pkl")
    joblib.dump({
        "threshold": optimal_thresh, "target_recall": target_recall,
        "model_name": best_name, "pr_auc": best_score
    }, f"{output_dir}/config.pkl")
    
    print(f"\nModel artifacts saved to: {output_dir}")
    
    return best_model, scaler, optimal_thresh, pd.DataFrame(results).sort_values("PR-AUC", ascending=False)

# -------------------- Prediction Function --------------------
def predict_fraud(new_data, model_dir="/mnt/data/fraud_model_final"):
    model = joblib.load(f"{model_dir}/model.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    config = joblib.load(f"{model_dir}/config.pkl")
    
    X_scaled = scaler.transform(new_data)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= config["threshold"]).astype(int)
    
    return preds, probs
# Run the pipeline
if __name__ == "__main__":
    FILE_PATH = r"F:\Python\Batch-2_LLM-Backend-B\data\processed\bfsi_cleaned_transactions.csv"

    
    print("🚀 Simple Optimized Fraud Detection")
    print("="*50)
    
    try:
        model, scaler, threshold, results = train_fraud_model(
            FILE_PATH, 
            target_recall=0.80  # Catch 80% of fraud cases
        )
        print("\n Success! Model ready for production.")
        
    except FileNotFoundError:
        print(f" File not found: {FILE_PATH}")
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()