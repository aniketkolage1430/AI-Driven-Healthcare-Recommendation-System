"""
AI-Driven Healthcare Recommendation System

Run:
    python app.py

This app trains several ML models from datasets/Training.csv, selects the best
model, saves it in models/health_model.pkl, and serves a modern Flask dashboard.
Predictions are educational and should not replace professional medical care.
"""

from __future__ import annotations

import ast
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
MODEL_PATH = BASE_DIR / "models" / "health_model.pkl"

app = Flask(__name__)


SYMPTOM_SYNONYMS = {
    "fever": "high_fever",
    "body pain": "muscle_pain",
    "body_pain": "muscle_pain",
    "weakness": "fatigue",
    "heart burn": "acidity",
    "heartburn": "acidity",
    "fire burn": "burning_micturition",
    "burning urine": "burning_micturition",
    "skin allergy": "skin_rash",
    "stomach ache": "stomach_pain",
    "loose motion": "diarrhoea",
    "yellow eyes": "yellowing_of_eyes",
}


AI_TIP_BANK = [
    "Drink enough water and monitor whether symptoms change over the next 24 hours.",
    "Avoid self-medication when symptoms are severe, recurring, or unusual.",
    "Track fever, pain level, appetite, sleep, and medication timing in a small note.",
    "Choose light meals and rest if you feel feverish, dizzy, nauseated, or weak.",
    "Seek urgent care for chest pain, breathing difficulty, fainting, confusion, or severe dehydration.",
]


def normalize_text(value: str) -> str:
    """Convert user text and dataset labels into comparable symptom keys."""
    value = value.strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_()]+", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def split_symptom_text(raw_text: str) -> list[str]:
    """Split comma/newline/free-text symptoms into clean phrases."""
    pieces = re.split(r"[,;\n]+| and | with ", raw_text.lower())
    return [piece.strip() for piece in pieces if piece.strip()]


def parse_list_like(value: Any) -> list[str]:
    """Read CSV cells that contain Python-list strings or plain text."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [item.strip() for item in re.split(r",|\|", text) if item.strip()]


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load features and target from the training dataset."""
    training_path = DATASET_DIR / "Training.csv"
    if not training_path.exists():
        raise FileNotFoundError("datasets/Training.csv is missing.")

    df = pd.read_csv(training_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "prognosis" not in df.columns:
        raise ValueError("Training.csv must contain a prognosis column.")

    x = df.drop(columns=["prognosis"])
    y = df["prognosis"]
    return x, y


def train_and_save_model() -> dict[str, Any]:
    """Train multiple models, select the highest accuracy model, and persist it."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    x, y = load_training_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=220, random_state=42, class_weight="balanced"
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(probability=True, kernel="rbf", C=2.0, random_state=42),
    }

    results: dict[str, float] = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results[name] = round(float(accuracy_score(y_test, predictions)), 4)
        fitted_models[name] = model

    best_model_name = max(results, key=results.get)
    best_model = fitted_models[best_model_name]
    best_predictions = best_model.predict(x_test)

    payload = {
        "model": best_model,
        "model_name": best_model_name,
        "accuracy": results[best_model_name],
        "model_scores": results,
        "features": list(x.columns),
        "classes": list(best_model.classes_),
        "classification_report": classification_report(
            y_test, best_predictions, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, best_predictions).tolist(),
    }

    with MODEL_PATH.open("wb") as file:
        pickle.dump(payload, file)

    return payload


def load_model_payload() -> dict[str, Any]:
    """Load the saved ML model, training it automatically if needed."""
    if not MODEL_PATH.exists():
        return train_and_save_model()
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


MODEL_PAYLOAD = load_model_payload()
FEATURES = MODEL_PAYLOAD["features"]
FEATURE_LOOKUP = {normalize_text(feature): feature for feature in FEATURES}


def build_input_frame(symptoms: list[str]) -> pd.DataFrame:
    """Create a one-row DataFrame with the exact training feature columns."""
    frame = pd.DataFrame([[0] * len(FEATURES)], columns=FEATURES)
    for symptom in symptoms:
        if symptom in FEATURES:
            frame.loc[0, symptom] = 1
    return frame


def map_user_symptoms(raw_text: str) -> tuple[list[str], list[str]]:
    """Map user-friendly symptom text to dataset feature names."""
    matched = []
    unknown = []

    for phrase in split_symptom_text(raw_text):
        synonym = SYMPTOM_SYNONYMS.get(phrase, phrase)
        normalized = normalize_text(synonym)

        if normalized in FEATURE_LOOKUP:
            matched.append(FEATURE_LOOKUP[normalized])
            continue

        partial_match = next(
            (feature for key, feature in FEATURE_LOOKUP.items() if normalized in key or key in normalized),
            None,
        )
        if partial_match:
            matched.append(partial_match)
        else:
            unknown.append(phrase)

    return sorted(set(matched)), unknown


def disease_rule_scores(symptoms: list[str]) -> dict[str, float]:
    """Score diseases by how strongly matched symptoms appear in their rows."""
    if not symptoms:
        return {}

    x, y = load_training_data()
    disease_scores = defaultdict(float)
    grouped = x.assign(prognosis=y).groupby("prognosis")

    for disease, rows in grouped:
        score = 0.0
        for symptom in symptoms:
            if symptom in rows.columns:
                score += float(rows[symptom].mean())
        if score > 0:
            disease_scores[disease] = score / max(len(symptoms), 1)

    return dict(disease_scores)


def smart_fallback_scores(raw_text: str) -> dict[str, float]:
    """Provide realistic fallback direction when free text has weak matches."""
    text = raw_text.lower()
    rules = {
        "Common Cold": ["cough", "sneeze", "runny", "throat", "congestion"],
        "GERD": ["acid", "heart", "burn", "indigestion"],
        "Migraine": ["headache", "light", "nausea", "vision"],
        "Urinary tract infection": ["urine", "burning", "bladder"],
        "Allergy": ["rash", "itch", "allergy", "sneeze"],
        "Malaria": ["fever", "chills", "sweating"],
    }

    scores = {}
    for disease, keywords in rules.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits:
            scores[disease] = hits / len(keywords)
    return scores


def top_predictions(raw_text: str, matched_symptoms: list[str]) -> list[dict[str, Any]]:
    """Blend ML probability, symptom rules, and fallback AI-like heuristics."""
    model = MODEL_PAYLOAD["model"]
    input_frame = build_input_frame(matched_symptoms)
    probabilities = model.predict_proba(input_frame)[0]
    ml_scores = dict(zip(model.classes_, probabilities))
    rule_scores = disease_rule_scores(matched_symptoms)
    fallback_scores = smart_fallback_scores(raw_text)

    all_diseases = set(ml_scores) | set(rule_scores) | set(fallback_scores)
    combined = {}
    for disease in all_diseases:
        combined[disease] = (
            (ml_scores.get(disease, 0.0) * 0.72)
            + (rule_scores.get(disease, 0.0) * 0.22)
            + (fallback_scores.get(disease, 0.0) * 0.06)
        )

    top_five = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:5]
    total = sum(score for _, score in top_five) or 1.0

    return [
        {
            "disease": disease,
            "percentage": round((score / total) * 100, 2),
            "raw_score": round(float(score), 5),
        }
        for disease, score in top_five
    ]


def load_recommendation_tables() -> dict[str, pd.DataFrame]:
    """Load all recommendation CSVs into DataFrames."""
    return {
        "description": pd.read_csv(DATASET_DIR / "description.csv"),
        "precautions": pd.read_csv(DATASET_DIR / "precautions_df.csv"),
        "medications": pd.read_csv(DATASET_DIR / "medications.csv"),
        "diets": pd.read_csv(DATASET_DIR / "diets.csv"),
        "workouts": pd.read_csv(DATASET_DIR / "workout_df.csv"),
    }


RECOMMENDATION_TABLES = load_recommendation_tables()


def recommendation_for(disease: str) -> dict[str, Any]:
    """Return description, medicines, diet, precautions, workout, and tips."""
    tables = RECOMMENDATION_TABLES
    disease_key = disease.strip().lower()

    description = "No detailed description is available for this disease in the dataset."
    desc_df = tables["description"]
    desc_row = desc_df[desc_df["Disease"].str.lower() == disease_key]
    if not desc_row.empty:
        description = str(desc_row.iloc[0]["Description"])

    precautions = []
    prec_df = tables["precautions"]
    prec_row = prec_df[prec_df["Disease"].str.lower() == disease_key]
    if not prec_row.empty:
        for column in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
            value = prec_row.iloc[0].get(column)
            if pd.notna(value):
                precautions.append(str(value).strip())

    medications = []
    med_df = tables["medications"]
    med_row = med_df[med_df["Disease"].str.lower() == disease_key]
    if not med_row.empty:
        medications = parse_list_like(med_row.iloc[0]["Medication"])

    diets = []
    diet_df = tables["diets"]
    diet_row = diet_df[diet_df["Disease"].str.lower() == disease_key]
    if not diet_row.empty:
        diets = parse_list_like(diet_row.iloc[0]["Diet"])

    workouts = []
    workout_df = tables["workouts"]
    if "disease" in workout_df.columns:
        workout_rows = workout_df[workout_df["disease"].str.lower() == disease_key]
        workouts = workout_rows["workout"].dropna().astype(str).head(6).tolist()

    return {
        "description": description,
        "precautions": precautions or ["Consult a qualified healthcare professional."],
        "medications": medications or ["Consult a doctor before taking any medicine."],
        "diets": diets or ["Eat balanced, light, freshly prepared meals."],
        "workouts": workouts or ["Take rest and resume activity gradually."],
        "tips": AI_TIP_BANK[:],
    }


@app.route("/")
def index():
    """Render the single-page dashboard."""
    report = MODEL_PAYLOAD["classification_report"]
    weighted = report.get("weighted avg", {})
    validation_summary = {
        "weighted_precision": round(weighted.get("precision", 0) * 100, 2),
        "weighted_recall": round(weighted.get("recall", 0) * 100, 2),
        "weighted_f1": round(weighted.get("f1-score", 0) * 100, 2),
        "confusion_preview": MODEL_PAYLOAD["confusion_matrix"][:8],
        "class_preview": MODEL_PAYLOAD["classes"][:8],
    }
    return render_template(
        "index.html",
        symptoms=sorted(set(FEATURES)),
        model_name=MODEL_PAYLOAD["model_name"],
        accuracy=round(MODEL_PAYLOAD["accuracy"] * 100, 2),
        model_scores=json.dumps(MODEL_PAYLOAD["model_scores"]),
        validation_summary=json.dumps(validation_summary),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Accept symptoms as JSON and return predictions plus recommendations."""
    try:
        payload = request.get_json(silent=True) or {}
        raw_symptoms = str(payload.get("symptoms", "")).strip()
        if not raw_symptoms:
            return jsonify({"error": "Please enter at least one symptom."}), 400

        matched_symptoms, unknown_symptoms = map_user_symptoms(raw_symptoms)
        if not matched_symptoms:
            return jsonify(
                {
                    "error": "No dataset symptoms matched your input.",
                    "unknown_symptoms": unknown_symptoms,
                    "suggestion": "Try inputs like fever, cough, acidity, headache, skin rash, body pain.",
                }
            ), 422

        predictions = top_predictions(raw_symptoms, matched_symptoms)
        primary_disease = predictions[0]["disease"]
        recommendation = recommendation_for(primary_disease)

        return jsonify(
            {
                "input": raw_symptoms,
                "matched_symptoms": matched_symptoms,
                "unknown_symptoms": unknown_symptoms,
                "predictions": predictions,
                "primary_disease": primary_disease,
                "confidence": predictions[0]["percentage"],
                "recommendation": recommendation,
                "model": {
                    "name": MODEL_PAYLOAD["model_name"],
                    "accuracy": round(MODEL_PAYLOAD["accuracy"] * 100, 2),
                    "scores": MODEL_PAYLOAD["model_scores"],
                    "confusion_matrix": MODEL_PAYLOAD["confusion_matrix"],
                    "classification_report": MODEL_PAYLOAD["classification_report"],
                },
                "disclaimer": "This is an educational AI project, not a medical diagnosis.",
            }
        )
    except Exception as exc:  # Keeps API failures readable for beginner debugging.
        return jsonify({"error": "Prediction failed.", "details": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
