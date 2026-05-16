"""
Flask API — Healthcare Chatbot Backend
Endpoints:
  POST /chat     → main conversation endpoint
  POST /predict  → raw ML prediction (for testing)
  GET  /health   → server health check
"""

import os, sys, pickle, json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

# make sibling packages importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from nlp.processor  import NLPProcessor
from nlp.emergency  import EmergencyEngine

app = Flask(__name__, static_folder="../frontend")

# ── Load ML artifacts ─────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")

with open(os.path.join(MODEL_DIR, "model.pkl"),    "rb") as f: clf      = pickle.load(f)
with open(os.path.join(MODEL_DIR, "encoder.pkl"),  "rb") as f: le       = pickle.load(f)
with open(os.path.join(MODEL_DIR, "symptoms.pkl"), "rb") as f: ALL_SYMS = pickle.load(f)

desc_df = pd.read_csv(os.path.join(DATA_DIR, "symptom_description.csv"))
DESC_MAP = dict(zip(desc_df["Disease"], desc_df["Description"]))

nlp    = NLPProcessor()
engine = EmergencyEngine()

# ── Session state (in-memory, single user for prototype) ──────────────────────
session = {
    "collected_symptoms": [],
    "turns": 0,
    "stage": "start",   # start | collecting | predicted
}

def reset_session():
    session["collected_symptoms"] = []
    session["turns"] = 0
    session["stage"] = "start"


# ── ML prediction helper ──────────────────────────────────────────────────────
def predict_disease(symptoms: list[str]) -> tuple[str, float]:
    """Return (disease_name, confidence_pct)."""
    row = {s: 0 for s in ALL_SYMS}
    for s in symptoms:
        if s in row:
            row[s] = 1
    X = pd.DataFrame([row])
    proba   = clf.predict_proba(X)[0]
    top_idx = int(np.argmax(proba))
    disease = le.inverse_transform([top_idx])[0]
    conf    = round(float(proba[top_idx]) * 100, 1)
    return disease, conf


# ── Conversational response builder ──────────────────────────────────────────
def build_response(user_input: str) -> dict:
    parsed  = nlp.process(user_input)
    intent  = parsed["intent"]
    new_sym = parsed["symptoms"]

    # ── Greeting ──────────────────────────────────────────────────────────────
    if intent == "greeting" and session["stage"] == "start":
        reset_session()
        return {
            "message": (
                "👋 Hello! I am your Healthcare Assistant.\n\n"
                "I can help you analyze your symptoms and detect potential health conditions.\n\n"
                "Please describe how you are feeling — for example:\n"
                "*\"I have a headache, high fever and I feel very tired.\"*\n\n"
                "⚠️ *Disclaimer: This chatbot is not a substitute for professional medical advice.*"
            ),
            "stage": "start",
            "symptoms_collected": [],
            "alert": None,
        }

    # ── Thanks / Farewell ─────────────────────────────────────────────────────
    if intent in ("thanks", "farewell"):
        reset_session()
        return {
            "message": "You're welcome! Take care and stay healthy. 😊 Feel free to return anytime.",
            "stage": "end",
            "symptoms_collected": [],
            "alert": None,
        }

    # ── Direct emergency keyword ──────────────────────────────────────────────
    if intent == "emergency":
        return {
            "message": (
                "🚨 **EMERGENCY DETECTED**\n\n"
                "Please call emergency services immediately:\n"
                "- **Nigeria**: 112 or 199\n"
                "- **International**: 911 / 999 / 112\n\n"
                "Do not wait — go to the nearest hospital now."
            ),
            "stage": "emergency",
            "symptoms_collected": session["collected_symptoms"],
            "alert": {"level": "EMERGENCY"},
        }


    # ── Detect "no more symptoms" phrases ────────────────────────────────────
    NO_MORE_PHRASES = [
        "no", "none", "no more", "i don't have", "i do not have",
        "nothing", "that's all", "thats all", "that is all",
        "no other", "no others", "nothing else", "i have no other",
        "just that", "only that", "that's it", "thats it",
        "nothing more", "no more symptoms", "i have none",
    ]
    user_lower = user_input.lower().strip()
    says_no_more = (
        any(user_lower == phrase for phrase in NO_MORE_PHRASES) or
        any(user_lower.startswith(phrase) for phrase in NO_MORE_PHRASES)
    )

    # If user says no more AND we already have at least 1 symptom, predict now
    if says_no_more and len(session["collected_symptoms"]) >= 1:
        total_syms = session["collected_symptoms"]
        disease, conf = predict_disease(total_syms)
        alert         = engine.evaluate(total_syms, disease)
        description   = DESC_MAP.get(disease, "Please consult a medical professional.")
        sym_display   = ", ".join(s.replace("_", " ") for s in total_syms)
        msg_parts = [
            f"Understood! Let me analyse what you have shared.\n\n",
            f"📋 **Symptoms Recorded:** {sym_display}\n",
            f"🔍 **Predicted Condition:** {disease} ({conf}% confidence)\n",
            f"📖 **About this condition:** {description}\n",
            f"\n{alert['message']}",
        ]
        if alert["red_flags"]:
            flags = ", ".join(s.replace("_", " ") for s in alert["red_flags"])
            msg_parts.append(f"\n⚡ **Critical symptoms detected:** {flags}")
        msg_parts.append("\n\n*Would you like to check more symptoms? Describe them or type 'bye' to end.*")
        session["stage"] = "predicted"
        return {
            "message":            "".join(msg_parts),
            "stage":              "predicted",
            "symptoms_collected": total_syms,
            "prediction": {
                "disease":     disease,
                "confidence":  conf,
                "description": description,
            },
            "alert": {
                "level":     alert["level"],
                "message":   alert["message"],
                "score":     alert["score"],
                "red_flags": alert["red_flags"],
            },
        }

    # ── Collect symptoms ──────────────────────────────────────────────────────
    for s in new_sym:
        if s not in session["collected_symptoms"]:
            session["collected_symptoms"].append(s)

    session["turns"] += 1
    total_syms = session["collected_symptoms"]

    # Need at least 2 symptoms for a reliable prediction
    if len(total_syms) < 2:
        missing = 2 - len(total_syms)
        sym_list = ", ".join(s.replace("_", " ") for s in total_syms) if total_syms else "none yet"
        return {
            "message": (
                f"I have noted: **{sym_list}**.\n\n"
                f"Could you describe {missing} more symptom(s) you are experiencing? "
                "The more details you give, the more accurate my analysis will be."
            ),
            "stage": "collecting",
            "symptoms_collected": total_syms,
            "alert": None,
        }

    # ── Run prediction ────────────────────────────────────────────────────────
    disease, conf = predict_disease(total_syms)
    alert         = engine.evaluate(total_syms, disease)
    description   = DESC_MAP.get(disease, "Please consult a medical professional.")
    sym_display   = ", ".join(s.replace("_", " ") for s in total_syms)

    # Build readable message
    msg_parts = [
        f"📋 **Symptoms Recorded:** {sym_display}\n",
        f"🔍 **Predicted Condition:** {disease} ({conf}% confidence)\n",
        f"📖 **About this condition:** {description}\n",
        f"\n{alert['message']}",
    ]
    if alert["red_flags"]:
        flags = ", ".join(s.replace("_", " ") for s in alert["red_flags"])
        msg_parts.append(f"\n⚡ **Critical symptoms detected:** {flags}")

    msg_parts.append(
        "\n\n*Would you like to check more symptoms? Describe them or type 'bye' to end.*"
    )

    session["stage"] = "predicted"

    return {
        "message":            "".join(msg_parts),
        "stage":              "predicted",
        "symptoms_collected": total_syms,
        "prediction": {
            "disease":    disease,
            "confidence": conf,
            "description": description,
        },
        "alert": {
            "level":     alert["level"],
            "message":   alert["message"],
            "score":     alert["score"],
            "red_flags": alert["red_flags"],
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "Random Forest", "diseases": len(le.classes_)})

@app.route("/chat", methods=["POST"])
def chat():
    data  = request.get_json(force=True)
    text  = data.get("message", "").strip()
    if not text:
        return jsonify({"error": "Empty message"}), 400
    response = build_response(text)
    return jsonify(response)

@app.route("/predict", methods=["POST"])
def predict():
    """Raw prediction endpoint — accepts list of canonical symptom names."""
    data     = request.get_json(force=True)
    symptoms = data.get("symptoms", [])
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400
    disease, conf = predict_disease(symptoms)
    alert         = engine.evaluate(symptoms, disease)
    return jsonify({
        "disease":    disease,
        "confidence": conf,
        "alert":      alert,
        "description": DESC_MAP.get(disease, ""),
    })

@app.route("/reset", methods=["POST"])
def reset():
    reset_session()
    return jsonify({"status": "session reset"})

if __name__ == "__main__":
    print("Starting Healthcare Chatbot API on http://localhost:5000")
    app.run(debug=True, port=5000)
