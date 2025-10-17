from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # backend/
MODEL_DIR = os.path.join(BASE_DIR, "fraud_model_final")

MODEL_DIR = os.path.join(BASE_DIR, "fraud_model_final")
# Vite outputs to "dist" by default
FRONTEND_BUILD = os.path.join(BASE_DIR, "..", "frontend", "dist")

# -------------------------------------------------------------------
# Flask App
# -------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

def load_artifact(fname):
    """Load model artifacts safely."""
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return joblib.load(path)

# Load artifacts
try:
    model = load_artifact("model.pkl")
    scaler = load_artifact("scaler.pkl")
    features = load_artifact("features.pkl")
    config = load_artifact("config.pkl")
    print("✅ All model artifacts loaded successfully.")
except Exception as e:
    print("❌ Error loading model artifacts:", e)
    model = scaler = features = config = None

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"message": "✅ Fraud Detection API running."})

@app.route("/api/features", methods=["GET"])
def get_features():
    if features is None:
        return jsonify({"error": "Features not available."}), 500
    return jsonify({"features": features})

@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    candidates = [
        os.path.join(BASE_DIR, "data", "processed", "transactions_processed.csv"),
        os.path.join(BASE_DIR, "data", "processed", "bfsi_cleaned_transactions.csv"),
        os.path.join(BASE_DIR, "data", "transactions.csv"),
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if csv_path:
        df = pd.read_csv(csv_path)
        records = df.to_dict(orient="records")
    else:
        records = []
    return jsonify(records)

def _preprocess_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    if 'kyc_verified' in df.columns:
        df['kyc_verified'] = df['kyc_verified'].map(
            {'yes': 1, 'no': 0, 'y': 1, 'n': 0, True: 1, False: 0}
        ).fillna(df['kyc_verified'])
    if 'timestamp' in df.columns and 'hour' in features:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['hour'] = df['timestamp'].dt.hour.fillna(0).astype(int)
        except Exception:
            pass
    return df

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or features is None:
        return jsonify({"error": "Model artifacts not loaded properly."}), 500

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Empty or invalid JSON input."}), 400
        df = pd.DataFrame(payload)
        df = _preprocess_input_df(df)
        df = df.reindex(columns=features, fill_value=0)
        X_scaled = scaler.transform(df.values)
        probs = model.predict_proba(X_scaled)[:, 1]
        threshold = config.get("threshold", 0.5) if isinstance(config, dict) else 0.5
        preds = (probs >= threshold).astype(int)
        return jsonify({
            "predictions": preds.tolist(),
            "probabilities": probs.round(4).tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# Serve React build in production
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
