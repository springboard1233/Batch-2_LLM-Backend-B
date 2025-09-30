# backend/app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os
import warnings
import sqlite3
import logging
import jwt
import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.exceptions import InconsistentVersionWarning

# -------------------------------------------------------------------
# Suppress scikit-learn version warnings
# -------------------------------------------------------------------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "fraud_model_final")
FRONTEND_BUILD = os.path.join(BASE_DIR, "..", "frontend", "dist")
DB_PATH = os.path.join(BASE_DIR, "transactions.db")
SECRET_KEY = "super-secret-key"  # 🔒 Change this in production!

# -------------------------------------------------------------------
# Flask App
# -------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------
# Allow only frontend origin
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------------------------------------------------------
# Database Init
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
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()

def save_transaction(user_id, record):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO transactions (user_id, data, predicted, probability) VALUES (?, ?, ?, ?)",
            (user_id, str(record), record.get("predicted"), record.get("probability"))
        )
        conn.commit()

def fetch_transactions(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT id, data, predicted, probability FROM transactions WHERE user_id=? ORDER BY id DESC",
            (user_id,)
        )
        rows = cur.fetchall()
        return [
            {"id": r[0], "data": r[1], "predicted": r[2], "probability": r[3]}
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
        return jsonify({"message": "✅ User registered successfully"})
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
    config = load_artifact("config.pkl")
    logging.info("✅ Model artifacts loaded successfully")
except Exception as e:
    logging.error("❌ Error loading model artifacts: %s", e)
    raise

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def _preprocess_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    if 'kyc_verified' in df.columns:
        df['kyc_verified'] = df['kyc_verified'].map(
            {'yes': 1, 'no': 0, 'y': 1, 'n': 0, True: 1, False: 0}
        ).fillna(0)
    if 'timestamp' in df.columns and 'hour' in features:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['hour'] = df['timestamp'].dt.hour.fillna(0).astype(int)
        except Exception:
            df['hour'] = 0
    return df

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

@app.route("/api/predict", methods=["POST"])
@token_required
def predict(user_id):
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Empty or invalid JSON input."}), 400

        df = pd.DataFrame(payload)
        df = _preprocess_input_df(df)

        # Ensure all expected features exist
        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df.reindex(columns=features, fill_value=0)

        # Scale & predict
        X_scaled = scaler.transform(df.values)
        probs = model.predict_proba(X_scaled)[:, 1]
        threshold = config.get("threshold", 0.5) if isinstance(config, dict) else 0.5
        preds = (probs >= threshold).astype(int)

        # Store results
        df_result = df.copy()
        df_result['predicted'] = preds
        df_result['probability'] = probs.round(4)

        results = df_result.to_dict(orient="records")
        for rec in results:
            save_transaction(user_id, rec)

        logging.info("🔍 Prediction request processed for %d records by user %d", len(results), user_id)

        return jsonify({
            "predictions": preds.tolist(),
            "probabilities": probs.round(4).tolist()
        })

    except Exception as e:
        logging.error("❌ Prediction error: %s", e)
        return jsonify({"error": str(e)}), 500

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
