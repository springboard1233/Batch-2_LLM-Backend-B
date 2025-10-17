# backend/app.py
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import warnings
import sqlite3
import logging
import jwt
import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.exceptions import InconsistentVersionWarning
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------
# 🧠 LLM Explanation (Google Gemini Integration)
# ------------------------------------------------------------
import google.generativeai as genai

# Configure Gemini API key from environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# -------------------------------------------------------------------
# Suppress scikit-learn version warnings
# -------------------------------------------------------------------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# -------------------------------------------------------------------
# Config - OPTIMIZED for better accuracy
# -------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "fraud_model_final")
FRONTEND_BUILD = os.path.join(BASE_DIR, "..", "frontend", "dist")
DB_PATH = os.path.join(BASE_DIR, "transactions.db")
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key")

# TUNED: More balanced weighting between ML and rules
MODEL_WEIGHT = float(os.getenv("MODEL_WEIGHT", 0.6))
ALERT_RISK_THRESHOLD = float(os.getenv("ALERT_RISK_THRESHOLD", 0.45))

# LLM Configuration (for explainability)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"

# -------------------------------------------------------------------
# Flask App
# -------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://localhost:5174"]}})

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------------------------------------------------------
# Database Init (SQLite)
# -------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                data TEXT,
                predicted INTEGER,
                probability REAL,
                actual_label INTEGER DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fraud_alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER,
                customer_id TEXT,
                risk_score REAL,
                reason TEXT,
                ml_prob REAL,
                rule_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transaction_id) REFERENCES transactions(id)
            )
        """)
        
        try:
            conn.execute("ALTER TABLE transactions ADD COLUMN actual_label INTEGER DEFAULT NULL")
        except sqlite3.OperationalError:
            pass
            
        conn.commit()

def save_transaction(user_id, record):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO transactions (user_id, data, predicted, probability) VALUES (?, ?, ?, ?)",
            (user_id, str(record), int(record.get("predicted", 0)), float(record.get("probability", 0.0)))
        )
        conn.commit()
        return cur.lastrowid

def fetch_transactions(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT id, data, predicted, probability, created_at FROM transactions WHERE user_id=? ORDER BY id DESC",
            (user_id,)
        )
        rows = cur.fetchall()
        return [
            {"id": r[0], "data": r[1], "predicted": r[2], "probability": r[3], "created_at": r[4]}
            for r in rows
        ]

init_db()

# -------------------------------------------------------------------
# Auth Helpers
# -------------------------------------------------------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[-1]
        if not token:
            return jsonify({"error": "Token missing!"}), 401
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            user_id = data["user_id"]
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401
        return f(user_id, *args, **kwargs)
    return decorated

# -------------------------------------------------------------------
# Auth Routes
# -------------------------------------------------------------------
@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    hashed_pw = generate_password_hash(password)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
        return jsonify({"message": "User registered successfully"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists"}), 400

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT id, password FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        if not user or not check_password_hash(user[1], password):
            return jsonify({"error": "Invalid username or password"}), 401

    token = jwt.encode(
        {"user_id": user[0], "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=6)},
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )
    return jsonify({"token": token})

@app.route("/api/auth/me", methods=["GET"])
@token_required
def me(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT id, username FROM users WHERE id=?", (user_id,))
        user = cur.fetchone()
        if not user:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"id": user[0], "username": user[1]})

# -------------------------------------------------------------------
# Load model artifacts
# -------------------------------------------------------------------
def load_artifact(fname):
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return joblib.load(path)

try:
    model = load_artifact("model.pkl")
    scaler = load_artifact("scaler.pkl")
    features = load_artifact("features.pkl")
    try:
        config = load_artifact("config.pkl")
    except Exception:
        config = {}
    logging.info("Model artifacts loaded successfully")
except Exception as e:
    logging.error("Error loading model artifacts: %s", e)
    raise

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def _preprocess_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    
    if 'transaction_amount' in df.columns:
        df['amount'] = df['transaction_amount']
        df = df.drop(columns=['transaction_amount'])
    
    for col in ["transaction_id", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].notna().any():
            df['hour'] = df['timestamp'].dt.hour
            df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        else:
            df['hour'] = 0
            df['is_weekend'] = 0
            df['is_night'] = 0
        df = df.drop(columns=['timestamp'])
    
    if 'amount' in df.columns:
        df['log_amount'] = pd.Series(df['amount']).apply(lambda x: np.log1p(x) if pd.notna(x) else 0)
        amount_95th = df['amount'].quantile(0.95) if len(df) > 20 else 50000
        df['high_amount'] = (df['amount'] > amount_95th).astype(int)
        df['high_amount_night'] = df['high_amount'] * df.get('is_night', 0)
    
    if 'kyc_verified' in df.columns:
        df['kyc_verified'] = df['kyc_verified'].map(
            {'yes': 1, 'no': 0, 'y': 1, 'n': 0, 'true': 1, 'false': 0, True: 1, False: 0, 1: 1, 0: 0}
        ).fillna(0)
    
    df['txn_last_24h'] = 0
    
    if 'channel' in df.columns:
        channel_cols = [f for f in features if f.startswith('channel_')]
        if len(channel_cols) > 0:
            channel_map = {0: 'Mobile', 1: 'POS', 2: 'ATM', 3: 'Web'}
            df['channel'] = df['channel'].map(channel_map).fillna('Mobile')
            df = pd.get_dummies(df, columns=['channel'], drop_first=True)
    
    df = df.fillna(0)
    return df

# -------------------------------------------------------------------
# Rule Engine
# -------------------------------------------------------------------
def apply_rules(record: dict) -> (float, list):
    reasons = []
    score = 0.0

    weights = {
        "very_high_amount": 0.45,
        "high_amount_no_kyc": 0.35,
        "odd_hour_high_amt": 0.30,
        "new_account_high": 0.40,
        "very_new_account": 0.30,
        "weekend_high_amt": 0.20
    }

    txn_amt = float(record.get("transaction_amount", 0) or 0)
    account_age = int(record.get("account_age_days", 0))
    channel = int(record.get("channel", 0))
    kyc = int(record.get("kyc_verified", 0))
    
    hour = None
    is_weekend = False
    ts = record.get("timestamp")
    if ts:
        try:
            dt = pd.to_datetime(ts)
            hour = dt.hour
            is_weekend = dt.dayofweek >= 5
        except:
            pass
    
    if txn_amt > 500000:
        score += weights["very_high_amount"]
        reasons.append(f"Extremely high amount: ₹{txn_amt:,.0f}")
    elif txn_amt > 100000 and not kyc:
        score += weights["high_amount_no_kyc"]
        reasons.append(f"High-value transaction (₹{txn_amt:,.0f}) without KYC verification")
    
    if hour is not None and 2 <= hour <= 5 and txn_amt > 25000:
        score += weights["odd_hour_high_amt"]
        reasons.append(f"Large transaction (₹{txn_amt:,.0f}) during suspicious hours ({hour}:00)")
    
    if account_age < 30 and txn_amt > 100000:
        score += weights["new_account_high"]
        reasons.append(f"Very new account ({account_age} days old) with high transaction (₹{txn_amt:,.0f})")
    elif account_age < 7 and txn_amt > 50000:
        score += weights["very_new_account"]
        reasons.append(f"Extremely new account ({account_age} days old) with ₹{txn_amt:,.0f} transaction")
    
    if is_weekend and txn_amt > 200000:
        score += weights["weekend_high_amt"]
        reasons.append(f"Weekend transaction with high amount (₹{txn_amt:,.0f})")
    
    risk_factors = 0
    if txn_amt > 100000:
        risk_factors += 1
    if not kyc:
        risk_factors += 1
    if account_age < 60:
        risk_factors += 1
    if hour is not None and (hour < 6 or hour > 22):
        risk_factors += 1
    
    if risk_factors >= 3:
        score += 0.25
        reasons.append(f"Multiple risk factors present ({risk_factors} indicators)")
    
    return min(score, 1.0), reasons

# -------------------------------------------------------------------
# LLM Integration for Explainable AI
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# LLM Integration for Explainable AI - OPTIMIZED VERSION
# -------------------------------------------------------------------
def get_llm_explanation(transaction_data: dict, prediction: str, risk_score: float, 
                       ml_prob: float, rule_score: float, reasons: list, category: str) -> str:
    """Generate human-readable explanation using available LLM or fallback"""
    
    # Debug info
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.warning("❌ GOOGLE_API_KEY not found in environment")
        return generate_rule_based_explanation(prediction, risk_score, reasons, category)
    
    logging.info(f"✅ GOOGLE_API_KEY found, attempting Gemini explanation...")
    
    # Try Google Gemini
    try:
        explanation = generate_gemini_explanation(
            transaction_data, prediction, risk_score, ml_prob, rule_score, reasons, category
        )
        
        # Check if we got a real Gemini response or fallback
        if "mixed signals" in explanation.lower() or "risk score:" in explanation.lower():
            logging.warning("⚠️  Using rule-based fallback explanation")
        else:
            logging.info("✅ Using real Gemini AI explanation")
            
        return explanation
        
    except Exception as e:
        logging.warning(f"❌ Gemini explanation failed: {e}")
        return generate_rule_based_explanation(prediction, risk_score, reasons, category)

def generate_gemini_explanation(transaction_data: dict, prediction: str, risk_score: float, 
                              ml_prob: float, rule_score: float, reasons: list, category: str) -> str:
    """Generate explanation using Google Gemini"""
    try:
        # Use the working model name
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        
        prompt = f"""
        You are a fraud detection expert analyzing a banking transaction. Provide a clear, professional explanation.

        Transaction Details:
        - Amount: ₹{transaction_data.get('transaction_amount', 0):,.2f}
        - Account Age: {transaction_data.get('account_age_days', 0)} days
        - Channel: {['Mobile', 'POS', 'ATM', 'Web'][transaction_data.get('channel', 0)]}
        - KYC Verified: {'Yes' if transaction_data.get('kyc_verified') else 'No'}

        Model Analysis:
        - Prediction: {prediction}
        - Risk Score: {risk_score*100:.1f}%
        - ML Confidence: {ml_prob*100:.1f}%
        - Rule Engine Score: {rule_score*100:.1f}%
        - Category: {category}
        - Risk Factors: {', '.join(reasons) if reasons else 'None detected'}

        Provide a brief (3-4 sentences) professional explanation of why this transaction was classified as {prediction}. 
        Focus on the key risk indicators and what actions should be taken. Be specific and actionable.
        """

        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text.strip():
            logging.info("Successfully generated real Gemini explanation")
            return response.text.strip()
        else:
            logging.warning("Gemini returned empty response")
            return generate_rule_based_explanation(prediction, risk_score, reasons, category)
        
    except Exception as e:
        logging.warning(f" Gemini API error: {e}")
        return generate_rule_based_explanation(prediction, risk_score, reasons, category)

def generate_rule_based_explanation(prediction: str, risk_score: float, reasons: list, category: str) -> str:
    """Fallback rule-based explanation"""
    if prediction == "Fraud":
        if risk_score >= 0.8:
            severity = "very high risk"
            action = "Immediate verification required. Block transaction and contact customer."
        elif risk_score >= 0.6:
            severity = "high risk"
            action = "Manual review recommended. Verify with customer before processing."
        else:
            severity = "moderate risk"
            action = "Additional verification recommended before approval."
        
        reason_text = " Key concerns: " + "; ".join(reasons[:2]) if reasons else ""
        return f"This transaction shows {severity} of fraud (Risk Score: {risk_score*100:.0f}%).{reason_text} Action: {action}"
    
    elif prediction == "Suspicious":
        return f"This transaction shows mixed signals (Risk Score: {risk_score*100:.0f}%). Some risk indicators present but not conclusive. Recommend monitoring customer activity and consider additional verification for high-value transactions."
    
    else:
        if risk_score < 0.2:
            return f"This transaction appears safe with very low risk indicators (Risk Score: {risk_score*100:.0f}%). Customer profile and transaction pattern are normal. Proceed with standard processing."
        else:
            return f"This transaction is classified as legitimate but shows some minor risk indicators (Risk Score: {risk_score*100:.0f}%). Continue monitoring but no immediate action required."

def get_confidence_level(ml_prob: float, rule_score: float, risk_score: float) -> str:
    if risk_score >= 0.8:
        return "Very High"
    elif risk_score >= 0.6:
        return "High"
    elif risk_score >= 0.4:
        return "Medium"
    else:
        return "Low"

def categorize_transaction(ml_prob: float, rule_score: float, reasons: list) -> str:
    if rule_score > 0.7 and ml_prob < 0.3:
        return "Rule-Based Fraud"
    elif ml_prob > 0.7 and rule_score < 0.3:
        return "ML-Detected Anomaly"
    elif ml_prob > 0.5 and rule_score > 0.5:
        return "High-Confidence Fraud"
    elif any("amount" in reason.lower() for reason in reasons):
        return "Amount-Related Suspicion"
    elif any("kyc" in reason.lower() for reason in reasons):
        return "Verification Issue"
    elif any("hour" in reason.lower() or "weekend" in reason.lower() for reason in reasons):
        return "Timing Anomaly"
    elif any("account" in reason.lower() for reason in reasons):
        return "New Account Risk"
    else:
        return "General Suspicion"

def safe_apply_rules(record: dict) -> (float, list):
    try:
        return apply_rules(record)
    except Exception as e:
        logging.warning(f"Rule engine error: {e}")
        return 0.0, ["Rule evaluation failed - relying on ML model"]

def save_alert(transaction_db_id: int, customer_id: str, ml_prob: float, rule_score: float, risk_score: float, reasons: list):
    reason_text = "; ".join(reasons)[:512] if reasons else "ML Model Detection"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO fraud_alerts (transaction_id, customer_id, risk_score, reason, ml_prob, rule_score) VALUES (?, ?, ?, ?, ?, ?)",
            (transaction_db_id, customer_id, float(risk_score), reason_text, float(ml_prob), float(rule_score))
        )
        conn.commit()

# -------------------------------------------------------------------
# Protected Routes
# -------------------------------------------------------------------
@app.route("/api/features", methods=["GET"])
@token_required
def get_features(user_id):
    return jsonify({"features": features})

@app.route("/api/transactions", methods=["GET"])
@token_required
def get_transactions(user_id):
    return jsonify(fetch_transactions(user_id))

# -------------------------------------------------------------------
# Debug Routes (Add this here)
# -------------------------------------------------------------------
@app.route("/api/debug/models", methods=["GET"])
def list_available_models():
    """List all available Gemini models"""
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            return jsonify({"error": "GOOGLE_API_KEY not set"})
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        models = genai.list_models()
        
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description
                })
        
        return jsonify({"available_models": available_models})
    except Exception as e:
        return jsonify({"error": str(e)})
# -------------------------------------------------------------------
# -------------------------------------------------------------------    
@app.route("/api/test-gemini", methods=["GET"])
def test_gemini():
    """Test if Gemini API is working"""
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            return jsonify({"error": "GOOGLE_API_KEY not set"})
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        response = model.generate_content("Say 'Hello World' in 5 words or less.")
        
        return jsonify({
            "success": True,
            "response": response.text,
            "model_used": "gemini-2.0-flash-001"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------------------------------------------------------
# LLM Explanation Endpoint (Separate API for standalone LLM calls)
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Standalone LLM Explanation Endpoint
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Standalone LLM Explanation Endpoint - FIXED VERSION
# -------------------------------------------------------------------
@app.route("/api/llm_explanation", methods=["POST"])
@token_required
def llm_explanation_endpoint(user_id):
    """
    Endpoint to generate a human-readable explanation for a transaction using LLM.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        transaction_data = data.get("transaction_data", {})
        prediction = data.get("prediction", "Legit")
        risk_score = float(data.get("risk_score", 0.0))
        ml_prob = float(data.get("ml_prob", 0.0))
        rule_score = float(data.get("rule_score", 0.0))
        reasons = data.get("reasons", [])
        category = data.get("category", "General Suspicion")

        explanation = get_llm_explanation(
            transaction_data, prediction, risk_score,
            ml_prob, rule_score, reasons, category
        )

        return jsonify({"explanation": explanation})

    except Exception as e:
        logging.error(f"LLM explanation endpoint error: {e}")
        # Return a fallback explanation instead of error
        fallback_explanation = generate_rule_based_explanation(
            data.get("prediction", "Legit"), 
            float(data.get("risk_score", 0.0)),
            data.get("reasons", []),
            data.get("category", "General Suspicion")
        )
        return jsonify({"explanation": fallback_explanation})


# -------------------------------------------------------------------
# /api/predict
# -------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
@token_required
def predict(user_id):
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Empty or invalid JSON input."}), 400

        if isinstance(payload, dict):
            records = [payload]
        elif isinstance(payload, list):
            records = payload
        else:
            return jsonify({"error": "Payload must be a dict or list of dicts."}), 400

        df = pd.DataFrame(records)
        df = _preprocess_input_df(df)

        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df.reindex(columns=features, fill_value=0)

        try:
            X_scaled = scaler.transform(df.values)
        except Exception as e:
            logging.error("Preprocessing/Scaling error: %s", e)
            return jsonify({"error": f"Preprocessing failed: {e}"}), 400

        probs = model.predict_proba(X_scaled)[:, 1]
        threshold = config.get("threshold", 0.5) if isinstance(config, dict) else 0.5
        preds = (probs >= threshold).astype(int)

        results = []
        for i, rec in df.iterrows():
            rec_dict = rec.to_dict()
            rec_dict["predicted"] = int(preds[i])
            rec_dict["probability"] = float(round(probs[i], 6))

            transaction_db_id = save_transaction(user_id, rec_dict)
            ml_prob = float(probs[i])

            raw_rec = records[i] if i < len(records) else rec_dict
            rule_score, reasons = safe_apply_rules(raw_rec)

            if ml_prob > 0.8 or ml_prob < 0.2:
                combined = 0.8 * ml_prob + 0.2 * rule_score
            else:
                combined = MODEL_WEIGHT * ml_prob + (1.0 - MODEL_WEIGHT) * rule_score
            
            risk_score = float(round(combined, 4))

            is_ml_fraud = (ml_prob >= threshold)
            is_rule_fraud = (rule_score >= 0.6)
            is_combined_fraud = (risk_score >= ALERT_RISK_THRESHOLD)

            if is_combined_fraud or (is_ml_fraud and is_rule_fraud):
                final_label = "Fraud"
                confidence = "High"
            elif is_ml_fraud or is_rule_fraud:
                final_label = "Suspicious"
                confidence = "Medium"
            else:
                final_label = "Legit"
                confidence = "Low"
            
            confidence_level = get_confidence_level(ml_prob, rule_score, risk_score)
            category = categorize_transaction(ml_prob, rule_score, reasons)
            
            # Generate LLM explanation
            llm_explanation = get_llm_explanation(
                raw_rec, final_label, risk_score, ml_prob, rule_score, reasons, category
            )
            
            if final_label == "Fraud":
                if reasons:
                    message = "🚨 High risk detected: " + "; ".join(reasons)
                elif ml_prob >= 0.75:
                    message = "🤖 ML model detected high fraud probability"
                else:
                    message = "⚠️ Suspicious activity detected"
            elif final_label == "Suspicious":
                message = "⚠️ Suspicious - requires review: " + ("; ".join(reasons) if reasons else "Pattern anomaly detected")
            else:
                if ml_prob < 0.1 and rule_score < 0.1:
                    message = "✅ Transaction appears safe"
                elif ml_prob < 0.3:
                    message = "✓ Transaction appears legitimate"
                else:
                    message = "ℹ️ Low risk - monitor if unusual"

            if final_label == "Fraud":
                try:
                    save_alert(transaction_db_id, raw_rec.get("customer_id", None), ml_prob, rule_score, risk_score, reasons)
                except Exception as e:
                    logging.warning("Failed to save alert: %s", e)

            results.append({
                "transaction_id": raw_rec.get("transaction_id", f"tx_{transaction_db_id}"),
                "db_transaction_id": transaction_db_id,
                "prediction": final_label,
                "risk_score": risk_score,
                "ml_probability": ml_prob,
                "rule_score": rule_score,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "category": category,
                "reasons": reasons,
                "message": message,
                "explanation": llm_explanation,
                "breakdown": {
                    "ml_contribution": float(round(MODEL_WEIGHT * ml_prob, 4)),
                    "rules_contribution": float(round((1.0 - MODEL_WEIGHT) * rule_score, 4))
                }
            })

        logging.info("Prediction processed for %d records by user %d", len(results), user_id)
        return jsonify({"results": results})

    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# Transaction Labeling
# -------------------------------------------------------------------
@app.route("/api/transactions/<int:txn_id>/label", methods=["POST"])
@token_required
def label_transaction(user_id, txn_id):
    data = request.get_json()
    label = data.get("label")

    if label not in [0, 1]:
        return jsonify({"error": "Invalid label value. Must be 0 or 1"}), 400

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT user_id FROM transactions WHERE id=?", (txn_id,))
        txn = cur.fetchone()
        if not txn:
            return jsonify({"error": "Transaction not found"}), 404
        if txn[0] != user_id:
            return jsonify({"error": "Unauthorized access"}), 403

        conn.execute("UPDATE transactions SET actual_label=? WHERE id=?", (label, txn_id))
        conn.commit()

    return jsonify({"message": f"Transaction {txn_id} labeled as {'Fraud' if label == 1 else 'Legit'}"})


# -------------------------------------------------------------------
# Metrics Endpoint
# -------------------------------------------------------------------
@app.route("/api/metrics", methods=["GET"])
@token_required
def get_model_metrics(user_id):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT predicted, probability, actual_label FROM transactions WHERE user_id=? AND actual_label IS NOT NULL",
            (user_id,)
        )
        labeled_rows = cur.fetchall()

    if labeled_rows and len(labeled_rows) > 0:
        y_pred = np.array([r[0] for r in labeled_rows])
        y_prob = np.array([r[1] for r in labeled_rows])
        y_true = np.array([r[2] for r in labeled_rows])

        try:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            auc = 0
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_prob)
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0

            return jsonify({
                "accuracy": float(round(acc, 4)),
                "precision": float(round(prec, 4)),
                "recall": float(round(rec, 4)),
                "f1Score": float(round(f1, 4)),
                "truePositives": int(tp),
                "trueNegatives": int(tn),
                "falsePositives": int(fp),
                "falseNegatives": int(fn),
                "auc": float(round(auc, 4)),
                "labeledCount": len(labeled_rows),
                "metricsType": "real",
                "message": f"Based on {len(labeled_rows)} reviewed transactions"
            })
        except Exception as e:
            logging.error(f"Real metrics calculation error: {e}")

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT predicted, probability FROM transactions WHERE user_id=?",
            (user_id,)
        )
        all_rows = cur.fetchall()

    if not all_rows:
        return jsonify({
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1Score": 0,
            "truePositives": 0,
            "trueNegatives": 0,
            "falsePositives": 0,
            "falseNegatives": 0,
            "auc": 0,
            "labeledCount": 0,
            "metricsType": "none",
            "message": "No transactions available yet"
        })

    y_pred = np.array([r[0] for r in all_rows])
    y_prob = np.array([r[1] for r in all_rows])
    y_true = y_pred

    try:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    except Exception as e:
        acc = prec = rec = f1 = auc = 0
        tn = fp = fn = tp = 0

    return jsonify({
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1Score": float(round(f1, 4)),
        "truePositives": int(tp),
        "trueNegatives": int(tn),
        "falsePositives": int(fp),
        "falseNegatives": int(fn),
        "auc": float(round(auc, 4)),
        "labeledCount": 0,
        "metricsType": "self_consistency",
        "message": f"Based on {len(all_rows)} total transactions (self-consistency check)"
    })

# -------------------------------------------------------------------
# Alerts Endpoint
# -------------------------------------------------------------------
@app.route("/api/alerts", methods=["GET"])
@token_required
def get_alerts(user_id):
    limit = int(request.args.get("limit", 100))
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT alert_id, transaction_id, customer_id, risk_score, reason, ml_prob, rule_score, created_at "
            "FROM fraud_alerts ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = cur.fetchall()
    alerts = []
    for r in rows:
        alerts.append({
            "alert_id": r[0],
            "transaction_id": r[1],
            "customer_id": r[2],
            "risk_score": r[3],
            "reason": r[4],
            "ml_prob": r[5],
            "rule_score": r[6],
            "created_at": r[7]
        })
    return jsonify({"alerts": alerts})

# -------------------------------------------------------------------
# Serve React frontend
# -------------------------------------------------------------------
if os.path.exists(FRONTEND_BUILD):
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        if path != "" and os.path.exists(os.path.join(FRONTEND_BUILD, path)):
            return send_from_directory(FRONTEND_BUILD, path)
        return send_from_directory(FRONTEND_BUILD, "index.html")

# -------------------------------------------------------------------
# Run server
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)