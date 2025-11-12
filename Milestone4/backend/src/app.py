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
import re
import json
import ast
from werkzeug.utils import secure_filename # <-- IMPORT FOR FILE UPLOADS

# --- IMPORTS FOR CHATBOT SOLUTION 1 ---
import base64
import io
from PIL import Image
# --- END NEW IMPORTS ---

# --- LANGCHAIN IMPORTS REMOVED ---

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

# --- ADDED CONFIGURATION FOR FILE UPLOADS ---
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# --- END OF NEW CONFIG ---

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
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # <-- CORRECT LOCATION

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
                kyc_document_path TEXT DEFAULT NULL,
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

        try:
            conn.execute("ALTER TABLE transactions ADD COLUMN kyc_document_path TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass # Column already exists

        conn.commit()

# --- REPLACED FUNCTION FOR FILE UPLOAD ---
def save_transaction(user_id, record):
    with sqlite3.connect(DB_PATH) as conn:
        kyc_path = record.get("kyc_document_path", None)
        cur = conn.execute(
            "INSERT INTO transactions (user_id, data, predicted, probability, kyc_document_path) VALUES (?, ?, ?, ?, ?)",
            (user_id, json.dumps(record), int(record.get("predicted", 0)), float(record.get("probability", 0.0)), kyc_path)
        )
        conn.commit()
        return cur.lastrowid
# --- END REPLACED FUNCTION ---

# --- UPDATED FUNCTION ---
def fetch_transactions(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, data, predicted, probability, created_at, kyc_document_path FROM transactions WHERE user_id=? ORDER BY id DESC",
            (user_id,)
        )
        rows = cur.fetchall()

    results = []
    for r in rows:
        data_val = r["data"]
        data_obj = {}
        try:
            data_obj = json.loads(data_val)
        except (json.JSONDecodeError, TypeError):
            try:
                data_obj = ast.literal_eval(data_val)
            except (ValueError, SyntaxError):
                data_obj = {"raw_data": str(data_val), "parse_error": True}

        results.append({
            "id": r["id"],
            "data": data_obj,
            "predicted": r["predicted"],
            "probability": r["probability"],
            "created_at": r["created_at"],
            "kyc_document_path": r["kyc_document_path"]
        })
    return results

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
        df['amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce').fillna(0)
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
        df['kyc_verified'] = pd.to_numeric(df['kyc_verified'], errors='coerce').fillna(0).astype(int)

    df['txn_last_24h'] = 0

    if 'channel' in df.columns:
        channel_cols = [f for f in features if f.startswith('channel_')]
        if len(channel_cols) > 0:
            channel_map = {0: 'Mobile', 1: 'POS', 2: 'ATM', 3: 'Web'}
            df['channel'] = pd.to_numeric(df['channel'], errors='coerce').fillna(0).map(channel_map).fillna('Mobile')
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

    try:
        txn_amt = float(record.get("transaction_amount", 0) or 0)
    except (ValueError, TypeError):
        txn_amt = 0.0

    try:
        account_age = int(record.get("account_age_days", 0) or 0)
    except (ValueError, TypeError):
        account_age = 0

    try:
        channel = int(record.get("channel", 0) or 0)
    except (ValueError, TypeError):
        channel = 0

    try:
        kyc = int(record.get("kyc_verified", 0))
    except (ValueError, TypeError):
        kyc = 0


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

def parse_amount_from_reasons(reasons: list, default_amount: float = 0.0) -> float:
    # (Function unchanged)
    amount_pattern = re.compile(r"₹([\d,]+(?:\.\d+)?)")
    for reason in reasons:
        match = amount_pattern.search(reason)
        if match:
            try:
                amount_str = match.group(1).replace(",", "")
                return float(amount_str)
            except ValueError:
                continue
    return default_amount

def parse_age_from_reasons(reasons: list, default_age: int = 0) -> int:
    # (Function unchanged)
    age_pattern = re.compile(r"\((\d+) days old\)")
    for reason in reasons:
        match = age_pattern.search(reason)
        if match:
            try:
                age_str = match.group(1)
                return int(age_str)
            except ValueError:
                continue
    return default_age

def parse_kyc_from_reasons(reasons: list, default_kyc: bool = True) -> str:
    # (Function unchanged)
    is_verified = default_kyc
    for reason in reasons:
        reason_lower = reason.lower()
        if "without kyc" in reason_lower or "no kyc" in reason_lower:
            is_verified = False
            break
    return "Yes" if is_verified else "No"

def parse_datetime_from_reasons(reasons: list) -> str:
    # (Function unchanged)
    hour_info = None
    weekend_info = None
    hour_pattern = re.compile(r"during suspicious hours \((\d+):00\)")
    for reason in reasons:
        reason_lower = reason.lower()
        hour_match = hour_pattern.search(reason)
        if hour_match:
            hour_info = f"Around {hour_match.group(1)}:00"
        if "weekend transaction" in reason_lower:
            weekend_info = "During the weekend"
    if hour_info and weekend_info:
        return f"{weekend_info}, {hour_info.lower()}"
    if hour_info:
        return hour_info
    if weekend_info:
        return weekend_info
    return "N/A"

def get_llm_explanation(transaction_data: dict, prediction: str, risk_score: float,
                        ml_prob: float, rule_score: float, reasons: list, category: str) -> str:
    # (Function unchanged)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.warning("❌ GOOGLE_API_KEY not found in environment")
        return generate_rule_based_explanation(prediction, risk_score, reasons, category)
    logging.info(f"✅ GOOGLE_API_KEY found, attempting Gemini explanation...")
    try:
        explanation = generate_gemini_explanation(
            transaction_data, prediction, risk_score, ml_prob, rule_score, reasons, category
        )
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
    # (Function unchanged)
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        amount = transaction_data.get('transaction_amount')
        if not amount:
            amount = parse_amount_from_reasons(reasons, default_amount=0.0)
        try: amount = float(amount)
        except (ValueError, TypeError): amount = 0.0

        account_age = transaction_data.get('account_age_days')
        if account_age is None:
            account_age = parse_age_from_reasons(reasons, default_age=0)
        try: account_age = int(account_age)
        except (ValueError, TypeError): account_age = 0

        channel_index = transaction_data.get('channel')
        if channel_index is None: channel_index = 0
        try: channel = ['Mobile', 'POS', 'ATM', 'Web'][int(channel_index)]
        except (IndexError, ValueError, TypeError): channel = 'Mobile'

        kyc_val = transaction_data.get('kyc_verified')
        if kyc_val in [True, 1, '1']: kyc = 'Yes'
        else: kyc = 'No'

        timestamp_val = transaction_data.get('timestamp')
        if timestamp_val:
            try: timestamp = pd.to_datetime(timestamp_val).strftime('%Y-%m-%d %H:%M:%S')
            except Exception: timestamp = str(timestamp_val)
        else: timestamp = parse_datetime_from_reasons(reasons)

        prompt = f"""
        You are a fraud detection expert analyzing a banking transaction. Provide a clear, professional explanation.

        Transaction Details:
        - Amount: ₹{amount:,.2f}
        - Account Age: {account_age} days
        - Channel: {channel}
        - KYC Verified: {kyc}
        - Timestamp: {timestamp}

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
    # (Function unchanged)
    if prediction == "Fraud":
        if risk_score >= 0.8: severity, action = "very high risk", "Immediate verification required. Block transaction and contact customer."
        elif risk_score >= 0.6: severity, action = "high risk", "Manual review recommended. Verify with customer before processing."
        else: severity, action = "moderate risk", "Additional verification recommended before approval."
        reason_text = " Key concerns: " + "; ".join(reasons[:2]) if reasons else ""
        return f"This transaction shows {severity} of fraud (Risk Score: {risk_score*100:.0f}%).{reason_text} Action: {action}"
    elif prediction == "Suspicious":
        return f"This transaction shows mixed signals (Risk Score: {risk_score*100:.0f}%). Some risk indicators present but not conclusive. Recommend monitoring customer activity and consider additional verification for high-value transactions."
    else:
        if risk_score < 0.2: return f"This transaction appears safe with very low risk indicators (Risk Score: {risk_score*100:.0f}%). Customer profile and transaction pattern are normal. Proceed with standard processing."
        else: return f"This transaction is classified as legitimate but shows some minor risk indicators (Risk Score: {risk_score*100:.0f}%). Continue monitoring but no immediate action required."

def get_confidence_level(ml_prob: float, rule_score: float, risk_score: float) -> str:
    # (Function unchanged)
    if risk_score >= 0.8: return "Very High"
    elif risk_score >= 0.6: return "High"
    elif risk_score >= 0.4: return "Medium"
    else: return "Low"

def categorize_transaction(ml_prob: float, rule_score: float, reasons: list) -> str:
    # (Function unchanged)
    if rule_score > 0.7 and ml_prob < 0.3: return "Rule-Based Fraud"
    elif ml_prob > 0.7 and rule_score < 0.3: return "ML-Detected Anomaly"
    elif ml_prob > 0.5 and rule_score > 0.5: return "High-Confidence Fraud"
    elif any("amount" in reason.lower() for reason in reasons): return "Amount-Related Suspicion"
    elif any("kyc" in reason.lower() for reason in reasons): return "Verification Issue"
    elif any("hour" in reason.lower() or "weekend" in reason.lower() for reason in reasons): return "Timing Anomaly"
    elif any("account" in reason.lower() for reason in reasons): return "New Account Risk"
    else: return "General Suspicion"

def safe_apply_rules(record: dict) -> (float, list):
    # (Function unchanged)
    try: return apply_rules(record)
    except Exception as e: logging.warning(f"Rule engine error: {e}"); return 0.0, ["Rule evaluation failed - relying on ML model"]

def save_alert(transaction_db_id: int, customer_id: str, ml_prob: float, rule_score: float, risk_score: float, reasons: list):
    # (Function unchanged)
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
    # (Function unchanged)
    return jsonify({"features": features})

@app.route("/api/transactions", methods=["GET"])
@token_required
def get_transactions(user_id):
    # (Function unchanged)
    return jsonify(fetch_transactions(user_id))

# -------------------------------------------------------------------
# Debug Routes
# -------------------------------------------------------------------
@app.route("/api/debug/models", methods=["GET"])
def list_available_models():
    # (Function unchanged)
    try:
        if not os.getenv("GOOGLE_API_KEY"): return jsonify({"error": "GOOGLE_API_KEY not set"})
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        models = genai.list_models()
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append({"name": model.name, "display_name": model.display_name, "description": model.description})
        return jsonify({"available_models": available_models})
    except Exception as e: return jsonify({"error": str(e)})

@app.route("/api/test-gemini", methods=["GET"])
def test_gemini():
    # (Function unchanged)
    try:
        if not os.getenv("GOOGLE_API_KEY"): return jsonify({"error": "GOOGLE_API_KEY not set"})
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        response = model.generate_content("Say 'Hello World' in 5 words or less.")
        return jsonify({"success": True, "response": response.text, "model_used": "gemini-2.0-flash-001"})
    except Exception as e: return jsonify({"error": str(e)})

# -------------------------------------------------------------------
# Standalone LLM Explanation Endpoint - FIXED VERSION
# -------------------------------------------------------------------
@app.route("/api/llm_explanation", methods=["POST"])
@token_required
def llm_explanation_endpoint(user_id):
    # (Function unchanged)
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "No data provided"}), 400
        transaction_data = data.get("transaction_data", {})
        prediction = data.get("prediction", "Legit")
        risk_score = float(data.get("risk_score", 0.0))
        ml_prob = float(data.get("ml_prob", 0.0))
        rule_score = float(data.get("rule_score", 0.0))
        reasons = data.get("reasons", [])
        category = data.get("category", "General Suspicion")
        explanation = get_llm_explanation(transaction_data, prediction, risk_score, ml_prob, rule_score, reasons, category)
        return jsonify({"explanation": explanation})
    except Exception as e:
        logging.error(f"LLM explanation endpoint error: {e}")
        fallback_explanation = generate_rule_based_explanation(data.get("prediction", "Legit"), float(data.get("risk_score", 0.0)), data.get("reasons", []), data.get("category", "General Suspicion"))
        return jsonify({"explanation": fallback_explanation})

# -------------------------------------------------------------------
# /api/predict (REPLACED FOR FILE UPLOAD)
# -------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
@token_required
def predict(user_id):
    # (Function unchanged - includes file handling)
    try:
        save_path = None
        if 'pan_card' not in request.files: logging.warning("No 'pan_card' file part in request")
        else:
            file = request.files['pan_card']
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                logging.info(f"File saved to {save_path}")
            else: logging.warning("File part present but no file selected")

        raw_rec = request.form.to_dict()

        if save_path: raw_rec['kyc_document_path'] = save_path; raw_rec['kyc_verified'] = 1
        else: raw_rec['kyc_verified'] = 0

        records = [raw_rec]
        df = pd.DataFrame(records)
        df_preprocessed = _preprocess_input_df(df)

        for col in features:
            if col not in df_preprocessed.columns: df_preprocessed[col] = 0
        df_preprocessed = df_preprocessed.reindex(columns=features, fill_value=0)

        try: X_scaled = scaler.transform(df_preprocessed.values)
        except Exception as e: logging.error("Preprocessing/Scaling error: %s", e); return jsonify({"error": f"Preprocessing failed: {e}"}), 400

        probs = model.predict_proba(X_scaled)[:, 1]
        threshold = config.get("threshold", 0.5) if isinstance(config, dict) else 0.5
        preds = (probs >= threshold).astype(int)

        results = []
        for i, rec in df_preprocessed.iterrows():
            raw_rec = records[i] if i < len(records) else {}
            rec_dict = rec.to_dict(); rec_dict.update(raw_rec)
            rec_dict["predicted"] = int(preds[i]); rec_dict["probability"] = float(round(probs[i], 6))
            transaction_db_id = save_transaction(user_id, rec_dict)
            ml_prob = float(probs[i])
            rule_score, reasons = safe_apply_rules(raw_rec)
            if ml_prob > 0.8 or ml_prob < 0.2: combined = 0.8 * ml_prob + 0.2 * rule_score
            else: combined = MODEL_WEIGHT * ml_prob + (1.0 - MODEL_WEIGHT) * rule_score
            risk_score = float(round(combined, 4))
            is_ml_fraud = (ml_prob >= threshold); is_rule_fraud = (rule_score >= 0.6); is_combined_fraud = (risk_score >= ALERT_RISK_THRESHOLD)
            if is_combined_fraud or (is_ml_fraud and is_rule_fraud): final_label, confidence = "Fraud", "High"
            elif is_ml_fraud or is_rule_fraud: final_label, confidence = "Suspicious", "Medium"
            else: final_label, confidence = "Legit", "Low"
            confidence_level = get_confidence_level(ml_prob, rule_score, risk_score)
            category = categorize_transaction(ml_prob, rule_score, reasons)
            llm_explanation = get_llm_explanation(raw_rec, final_label, risk_score, ml_prob, rule_score, reasons, category)
            if final_label == "Fraud":
                if reasons: message = "🚨 High risk detected: " + "; ".join(reasons)
                elif ml_prob >= 0.75: message = "🤖 ML model detected high fraud probability"
                else: message = "⚠️ Suspicious activity detected"
            elif final_label == "Suspicious": message = "⚠️ Suspicious - requires review: " + ("; ".join(reasons) if reasons else "Pattern anomaly detected")
            else:
                if ml_prob < 0.1 and rule_score < 0.1: message = "✅ Transaction appears safe"
                elif ml_prob < 0.3: message = "✓ Transaction appears legitimate"
                else: message = "ℹ️ Low risk - monitor if unusual"
            if final_label == "Fraud":
                try: save_alert(transaction_db_id, raw_rec.get("customer_id", None), ml_prob, rule_score, risk_score, reasons)
                except Exception as e: logging.warning("Failed to save alert: %s", e)
            results.append({
                "transaction_id": raw_rec.get("transaction_id", f"tx_{transaction_db_id}"), "db_transaction_id": transaction_db_id,
                "prediction": final_label, "risk_score": risk_score, "ml_probability": ml_prob, "rule_score": rule_score,
                "confidence": confidence, "confidence_level": confidence_level, "category": category, "reasons": reasons,
                "message": message, "explanation": llm_explanation,
                "breakdown": {"ml_contribution": float(round(MODEL_WEIGHT * ml_prob, 4)), "rules_contribution": float(round((1.0 - MODEL_WEIGHT) * rule_score, 4))}
            })
        logging.info("Prediction processed for %d records by user %d", len(results), user_id)
        return jsonify({"results": results})
    except Exception as e: logging.exception("Prediction error"); return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# Admin Analytics AI Assistant (LLM-Powered)
# -------------------------------------------------------------------
@app.route("/api/admin/analytics/query", methods=["POST"])
@token_required
def admin_analytics_query(user_id):
    """
    LLM-Powered AI Assistant for admin analytics - uses Gemini to understand
    natural language queries and generate dynamic SQL queries
    """
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Use LLM to analyze the query and generate appropriate SQL
        llm_response = analyze_query_with_llm(query)
        
        if "error" in llm_response:
            return jsonify({"error": llm_response["error"]}), 500

        # Execute the generated SQL and get results
        response = execute_analytics_query(llm_response)
        
        return jsonify(response)

    except Exception as e:
        logging.error(f"Admin analytics error: {e}")
        return jsonify({"error": str(e)}), 500

def analyze_query_with_llm(user_query):
    """Use Gemini to understand the user's query and generate appropriate SQL"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        
        prompt = f"""
        You are a data analyst for a fraud detection system. Analyze the user's query and determine what data they need.
        
        DATABASE SCHEMA:
        - transactions: id, user_id, data (JSON), predicted, probability, created_at
        - fraud_alerts: alert_id, transaction_id, customer_id, risk_score, reason, ml_prob, rule_score, created_at
        
        USER QUERY: "{user_query}"
        
        Based on the query, respond with JSON in this exact format:
        {{
            "query_type": "top_users|fraud_trends|channel_analysis|risk_distribution|account_age|kyc_status|system_overview",
            "time_period_days": 7,
            "sql_query": "the actual SQL query to run",
            "chart_type": "bar|line|pie|table",
            "explanation": "brief explanation of what this query will show"
        }}
        
        Available query_types:
        - top_users: Top fraud-prone users/customers
        - fraud_trends: Fraud patterns over time
        - channel_analysis: Fraud by transaction channel (Mobile, POS, ATM, Web)
        - risk_distribution: Distribution of risk scores
        - account_age: Fraud by account age groups
        - kyc_status: Fraud by KYC verification status
        - system_overview: General system statistics
        
        Generate appropriate SQL based on the user's natural language query.
        """
        
        response = model.generate_content(prompt)
        
        # Parse the LLM response as JSON
        try:
            import re
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Could not parse LLM response as JSON"}
        except json.JSONDecodeError:
            return {"error": "LLM response was not valid JSON"}
            
    except Exception as e:
        logging.error(f"LLM analysis error: {e}")
        return {"error": f"LLM analysis failed: {str(e)}"}

def execute_analytics_query(llm_response):
    """Execute the SQL query generated by LLM and format the response"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            # Execute the SQL query
            cursor = conn.execute(llm_response["sql_query"])
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            data = [dict(row) for row in rows]
            
            # Generate human-readable answer
            answer = generate_llm_answer(llm_response, data, len(rows))
            
            return {
                "answer": answer,
                "data": data,
                "chartType": llm_response["chart_type"],
                "queryType": llm_response["query_type"],
                "explanation": llm_response["explanation"]
            }
            
    except Exception as e:
        logging.error(f"Query execution error: {e}")
        return {
            "answer": f"Error executing query: {str(e)}",
            "data": [],
            "chartType": "table",
            "queryType": "error"
        }

def generate_llm_answer(llm_response, data, row_count):
    """Use LLM to generate a natural language answer from the data"""
    try:
        if row_count == 0:
            return "No data found matching your query."
        
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        
        prompt = f"""
        You are a data analyst presenting insights to a business user.
        
        QUERY TYPE: {llm_response['query_type']}
        EXPLANATION: {llm_response['explanation']}
        DATA (JSON): {json.dumps(data, indent=2)}
        NUMBER OF RESULTS: {row_count}
        
        Create a clear, concise 2-3 paragraph analysis of this data. Focus on:
        1. Key findings and patterns
        2. Important numbers or percentages  
        3. Business implications
        4. Any recommendations if relevant
        
        Write in professional but accessible language. Use bullet points if helpful.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        logging.error(f"LLM answer generation error: {e}")
        # Fallback to basic answer
        return f"Found {row_count} results for your query about {llm_response['query_type'].replace('_', ' ')}."

# -------------------------------------------------------------------
# Fallback Analytics (if LLM fails)
# -------------------------------------------------------------------
@app.route("/api/admin/analytics/simple-query", methods=["POST"])
@token_required
def admin_analytics_simple_query(user_id):
    """Fallback analytics without LLM - uses predefined queries"""
    try:
        data = request.get_json()
        query = data.get("query", "").strip().lower()
        
        if not query:
            return jsonify({"error": "Query is required"}), 400

        response = {
            "answer": "",
            "data": None,
            "chartType": None
        }

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            # Check for data
            total_alerts = conn.execute("SELECT COUNT(*) as count FROM fraud_alerts").fetchone()['count']
            total_transactions = conn.execute("SELECT COUNT(*) as count FROM transactions").fetchone()['count']
            
            if total_alerts == 0 and total_transactions == 0:
                response["answer"] = "No transaction data available yet. Submit some transactions to see analytics."
                return jsonify(response)
            
            # Simple keyword-based routing
            if any(word in query for word in ["top", "user", "customer"]):
                response = get_top_fraud_users_simple(conn, query)
            elif any(word in query for word in ["channel", "pos", "mobile", "atm", "web"]):
                response = get_fraud_by_channel_simple(conn)
            elif any(word in query for word in ["trend", "time", "week", "month"]):
                response = get_fraud_trends_simple(conn, query)
            elif any(word in query for word in ["risk", "score"]):
                response = get_risk_distribution_simple(conn)
            elif any(word in query for word in ["overview", "summary", "stat"]):
                response = get_system_overview_simple(conn)
            else:
                response["answer"] = "I can help you analyze: top users, fraud trends, risk scores, transaction channels, or system overview."

        return jsonify(response)

    except Exception as e:
        logging.error(f"Simple analytics error: {e}")
        return jsonify({"error": str(e)}), 500

def get_top_fraud_users_simple(conn, query):
    days = 7
    if "month" in query: days = 30
    
    sql = f"""
    SELECT customer_id, COUNT(*) as fraud_count, AVG(risk_score) as avg_risk
    FROM fraud_alerts 
    WHERE created_at >= datetime('now', '-{days} days')
    GROUP BY customer_id 
    ORDER BY fraud_count DESC 
    LIMIT 10
    """
    
    cursor = conn.execute(sql)
    rows = cursor.fetchall()
    data = [dict(row) for row in rows]
    
    answer = f"Top {len(data)} fraud-prone users (last {days} days):\n\n"
    for i, item in enumerate(data, 1):
        answer += f"{i}. {item['customer_id']}: {item['fraud_count']} alerts, avg risk: {item['avg_risk']:.1%}\n"
    
    return {"answer": answer, "data": data, "chartType": "bar"}

def get_fraud_trends_simple(conn, query):
    days = 30
    if "week" in query: days = 7
    
    sql = f"""
    SELECT DATE(created_at) as date, COUNT(*) as count, AVG(risk_score) as avg_risk
    FROM fraud_alerts 
    WHERE created_at >= datetime('now', '-{days} days')
    GROUP BY DATE(created_at) 
    ORDER BY date
    """
    
    cursor = conn.execute(sql)
    rows = cursor.fetchall()
    data = [dict(row) for row in rows]
    
    total = sum(item['count'] for item in data)
    answer = f"Fraud trends (last {days} days): {total} total alerts\nAverage daily: {total/max(len(data),1):.1f} alerts"
    
    return {"answer": answer, "data": data, "chartType": "line"}

def get_risk_distribution_simple(conn):
    sql = """
    SELECT 
        CASE 
            WHEN risk_score < 0.3 THEN 'Low'
            WHEN risk_score < 0.6 THEN 'Medium' 
            WHEN risk_score < 0.8 THEN 'High'
            ELSE 'Critical'
        END as risk_level,
        COUNT(*) as count
    FROM fraud_alerts 
    GROUP BY risk_level 
    ORDER BY count DESC
    """
    
    cursor = conn.execute(sql)
    rows = cursor.fetchall()
    data = [dict(row) for row in rows]
    
    answer = "Risk score distribution:\n"
    for item in data:
        answer += f"• {item['risk_level']}: {item['count']} alerts\n"
    
    return {"answer": answer, "data": data, "chartType": "pie"}

def get_system_overview_simple(conn):
    total_txns = conn.execute("SELECT COUNT(*) as count FROM transactions").fetchone()['count']
    fraud_txns = conn.execute("SELECT COUNT(*) as count FROM transactions WHERE predicted = 1").fetchone()['count']
    total_alerts = conn.execute("SELECT COUNT(*) as count FROM fraud_alerts").fetchone()['count']
    
    data = [
        {"metric": "Total Transactions", "value": total_txns},
        {"metric": "Fraud Detected", "value": fraud_txns},
        {"metric": "Fraud Alerts", "value": total_alerts}
    ]
    
    fraud_rate = (fraud_txns / total_txns * 100) if total_txns > 0 else 0
    answer = f"System Overview:\n• Transactions: {total_txns}\n• Fraud Rate: {fraud_rate:.1f}%\n• Alerts: {total_alerts}"
    
    return {"answer": answer, "data": data, "chartType": "bar"}

def get_fraud_by_channel_simple(conn):
    """Get fraud distribution by transaction channel"""
    sql = """
    SELECT 
        CASE 
            WHEN t.data LIKE '%channel%' THEN 
                CASE 
                    WHEN json_extract(t.data, '$.channel') = '0' THEN 'Mobile'
                    WHEN json_extract(t.data, '$.channel') = '1' THEN 'POS'
                    WHEN json_extract(t.data, '$.channel') = '2' THEN 'ATM'
                    WHEN json_extract(t.data, '$.channel') = '3' THEN 'Web'
                    ELSE 'Unknown'
                END
            ELSE 'Unknown'
        END as channel,
        COUNT(*) as fraud_count,
        AVG(f.risk_score) as avg_risk
    FROM fraud_alerts f
    JOIN transactions t ON f.transaction_id = t.id
    GROUP BY channel
    ORDER BY fraud_count DESC
    """
    
    cursor = conn.execute(sql)
    rows = cursor.fetchall()
    data = [dict(row) for row in rows]
    
    answer = "Fraud by transaction channel:\n"
    for item in data:
        answer += f"• {item['channel']}: {item['fraud_count']} alerts, avg risk: {item['avg_risk']:.1%}\n"
    
    return {"answer": answer, "data": data, "chartType": "pie"}
# --- END REPLACED FUNCTION ---


# -------------------------------------------------------------------
# Transaction Labeling
# -------------------------------------------------------------------
@app.route("/api/transactions/<int:txn_id>/label", methods=["POST"])
@token_required
def label_transaction(user_id, txn_id):
    # (Function unchanged)
    data = request.get_json()
    label = data.get("label")
    if label not in [0, 1]: return jsonify({"error": "Invalid label value. Must be 0 or 1"}), 400
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT user_id FROM transactions WHERE id=?", (txn_id,))
        txn = cur.fetchone()
        if not txn: return jsonify({"error": "Transaction not found"}), 404
        if txn[0] != user_id: return jsonify({"error": "Unauthorized access"}), 403
        conn.execute("UPDATE transactions SET actual_label=? WHERE id=?", (label, txn_id))
        conn.commit()
    return jsonify({"message": f"Transaction {txn_id} labeled as {'Fraud' if label == 1 else 'Legit'}"})


# -------------------------------------------------------------------
# Metrics Endpoint (REMOVED)
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Alerts Endpoint
# -------------------------------------------------------------------
@app.route("/api/alerts", methods=["GET"])
@token_required
def get_alerts(user_id):
    # (Function unchanged)
    limit = int(request.args.get("limit", 100))
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT alert_id, transaction_id, customer_id, risk_score, reason, ml_prob, rule_score, created_at "
            "FROM fraud_alerts ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = cur.fetchall()
    alerts = []
    for r in rows:
        alerts.append({
            "alert_id": r[0], "transaction_id": r[1], "customer_id": r[2], "risk_score": r[3],
            "reason": r[4], "ml_prob": r[5], "rule_score": r[6], "created_at": r[7]
        })
    return jsonify({"alerts": alerts})


# -------------------------------------------------------------------
# NEW CHATBOT ENDPOINT (SOLUTION 1)
# -------------------------------------------------------------------
@app.route("/api/chatbot/check_image", methods=["POST"])
@token_required
def check_kyc_image(user_id):
    # (Function added)
    data = request.get_json()
    if 'image' not in data: return jsonify({"error": "No image data found"}), 400
    try:
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        model = genai.GenerativeModel("gemini-pro-vision")
        prompt = """
        You are a helpful KYC verification assistant. Look at this image of a document.
        Provide brief, real-time feedback for the user in 1-2 sentences. Check for:
        1. Blurriness 2. Glare/Reflection 3. Cut Off edges 4. Expiry (if visible).
        - If good: "This looks clear and readable! Ready to submit."
        - If bad: Gently point out the problem (e.g., "This image appears a bit blurry... Can you try taking it in a brighter room?")
        """
        response = model.generate_content([prompt, image])
        return jsonify({"feedback": response.text})
    except Exception as e: logging.error(f"Image check error: {e}"); return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# CHATBOT ENDPOINT (SOLUTION 2 - REMOVED - NO LANGCHAIN)
# -------------------------------------------------------------------
# The /api/chatbot/assist endpoint was here and has been removed.


# -------------------------------------------------------------------
# Serve React frontend
# -------------------------------------------------------------------
if os.path.exists(FRONTEND_BUILD):
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        # (Function unchanged)
        if path != "" and os.path.exists(os.path.join(FRONTEND_BUILD, path)):
            return send_from_directory(FRONTEND_BUILD, path)
        return send_from_directory(FRONTEND_BUILD, "index.html")

# -------------------------------------------------------------------
# Run server
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)