"""
AgriSmart AI - Enhanced Flask Backend
Crop Recommendation System — upgraded with confidence scores,
top-5 alternatives, feature importance, prediction history, and crop info.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import requests
import re
from datetime import datetime
import sqlite3
import time
from collections import defaultdict

# ── Database Setup ─────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH   = os.path.join(BASE_DIR, "data", "chatbot.db")

def init_db():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chatbot_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Simple in-memory rate limiting dictionary
chat_rate_limits = defaultdict(list)

def is_rate_limited(user_id, max_requests=15, period=60):
    now = time.time()
    chat_rate_limits[user_id] = [t for t in chat_rate_limits[user_id] if now - t < period]
    if len(chat_rate_limits[user_id]) >= max_requests:
        return True
    chat_rate_limits[user_id].append(now)
    return False

def save_chat_to_db(user_id, user_message, bot_response):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chatbot_history (user_id, user_message, bot_response)
        VALUES (?, ?, ?)
    """, (user_id, user_message, bot_response))
    conn.commit()
    conn.close()

def get_chat_history_from_db(user_id, limit=20):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_message, bot_response, timestamp FROM chatbot_history
        WHERE user_id = ?
        ORDER BY timestamp ASC
        LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "user": row["user_message"],
            "bot": row["bot_response"],
            "timestamp": row["timestamp"]
        })
    return history

# ── App Setup ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER   = os.path.join(BASE_DIR, "static")
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
app = Flask(
    __name__,
    static_folder=STATIC_FOLDER,
    
    template_folder=TEMPLATE_FOLDER,
    static_url_path="/static"
)

MODEL_PATH  = os.path.join(BASE_DIR, "models", "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

prediction_history = []

# ── Crop Knowledge Base ────────────────────────────────────────────────────────
CROP_INFO = {
    "rice":        {"emoji": "🌾", "season": "Kharif", "water": "High",   "tip": "Grows best in waterlogged fields with high humidity."},
    "maize":       {"emoji": "🌽", "season": "Kharif", "water": "Medium", "tip": "Needs well-drained soil and moderate rainfall."},
    "chickpea":    {"emoji": "🫘", "season": "Rabi",   "water": "Low",    "tip": "Drought-tolerant; ideal for dry winters."},
    "kidneybeans": {"emoji": "🫘", "season": "Kharif", "water": "Medium", "tip": "Prefers loamy soil with good organic content."},
    "pigeonpeas":  {"emoji": "🌿", "season": "Kharif", "water": "Low",    "tip": "Highly drought-tolerant and nitrogen-fixing."},
    "mothbeans":   {"emoji": "🌱", "season": "Kharif", "water": "Low",    "tip": "Thrives in arid regions with sandy soil."},
    "mungbean":    {"emoji": "🟢", "season": "Kharif", "water": "Low",    "tip": "Quick-growing; good for crop rotation."},
    "blackgram":   {"emoji": "⚫", "season": "Kharif", "water": "Medium", "tip": "Fixes nitrogen in soil, improving fertility."},
    "lentil":      {"emoji": "🫘", "season": "Rabi",   "water": "Low",    "tip": "Thrives in cool, dry climates with well-drained soil."},
    "pomegranate": {"emoji": "🍎", "season": "Annual", "water": "Low",    "tip": "Drought-hardy; produces best in hot, dry summers."},
    "banana":      {"emoji": "🍌", "season": "Annual", "water": "High",   "tip": "Needs warm temperatures and rich, moist soil."},
    "mango":       {"emoji": "🥭", "season": "Summer", "water": "Medium", "tip": "Thrives in tropical climates; needs a dry flowering season."},
    "grapes":      {"emoji": "🍇", "season": "Annual", "water": "Medium", "tip": "Prefers well-drained, slightly acidic soil."},
    "watermelon":  {"emoji": "🍉", "season": "Summer", "water": "Medium", "tip": "Needs long warm days and sandy loam soil."},
    "muskmelon":   {"emoji": "🍈", "season": "Summer", "water": "Medium", "tip": "Grows well in hot, dry weather with light soil."},
    "apple":       {"emoji": "🍏", "season": "Winter", "water": "Medium", "tip": "Requires cold winters for dormancy and fruit set."},
    "orange":      {"emoji": "🍊", "season": "Winter", "water": "Medium", "tip": "Prefers subtropical climate with mild winters."},
    "papaya":      {"emoji": "🍈", "season": "Annual", "water": "Medium", "tip": "Fast-growing; dislikes waterlogging."},
    "coconut":     {"emoji": "🥥", "season": "Annual", "water": "High",   "tip": "Thrives in humid coastal regions with sandy soil."},
    "cotton":      {"emoji": "🌿", "season": "Kharif", "water": "Medium", "tip": "Requires long frost-free season and deep soil."},
    "jute":        {"emoji": "🌾", "season": "Kharif", "water": "High",   "tip": "Needs warm, humid climate with heavy rainfall."},
    "coffee":      {"emoji": "☕", "season": "Annual", "water": "Medium", "tip": "Grows best at high altitudes with well-distributed rainfall."},
}

def load_dotenv_from_file(env_path: str) -> None:
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

load_dotenv_from_file(os.path.join(BASE_DIR, ".env"))

WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY") or os.environ.get("OPENWEATHER_API_KEY")
WEATHER_BASE_URL = "http://api.weatherapi.com/v1/current.json"


def get_weather_for_location(location: str) -> dict:
    if not WEATHER_API_KEY:
        raise ValueError("Weather API key is not configured. Set WEATHER_API_KEY in your .env file.")

    params = {
        "key": WEATHER_API_KEY,
        "q": location
    }

    response = requests.get(WEATHER_BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()

    location_data = payload.get("location", {})
    current_data = payload.get("current", {})

    temperature = current_data.get("temp_c", 0.0)
    humidity = current_data.get("humidity", 0)
    # precip_mm represents current precipitation / rainfall
    rainfall = current_data.get("precip_mm", 0.0)

    return {
        "location_name": location_data.get("name", location),
        "temperature": round(float(temperature), 1),
        "humidity": int(humidity),
        "rainfall": round(float(rainfall), 1)
    }

# ── Load Model & Scaler ────────────────────────────────────────────────────────
try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}\n   Run 'python src/train_model.py' first.")
    model = scaler = None

# ── Load Dataset for Chatbot Statistics ───────────────────────────────────────
df_data = None
DATASET_DEFAULTS = {
    "N": 50.5,
    "P": 53.4,
    "K": 48.1,
    "temperature": 25.6,
    "humidity": 71.5,
    "ph": 6.5,
    "rainfall": 103.4
}

try:
    DATASET_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")
    if os.path.exists(DATASET_PATH):
        df_data = pd.read_csv(DATASET_PATH)
        print("Dataset loaded successfully for chatbot lookup.")
        try:
            DATASET_DEFAULTS = {
                "N": round(float(df_data["N"].mean()), 1),
                "P": round(float(df_data["P"].mean()), 1),
                "K": round(float(df_data["K"].mean()), 1),
                "temperature": round(float(df_data["temperature"].mean()), 1),
                "humidity": round(float(df_data["humidity"].mean()), 1),
                "ph": round(float(df_data["ph"].mean()), 1),
                "rainfall": round(float(df_data["rainfall"].mean()), 1)
            }
        except Exception as stats_err:
            print(f"Error computing dataset default averages: {stats_err}")
    else:
        print(f"Warning: Crop_recommendation.csv not found at {DATASET_PATH}")
except Exception as e:
    print(f"Warning: Could not load dataset for chatbot lookup: {e}")

def get_crop_stats(crop_name):
    if df_data is not None:
        crop_df = df_data[df_data["label"].str.lower() == crop_name.lower()]
        if not crop_df.empty:
            return {
                "N": round(float(crop_df["N"].mean()), 1),
                "P": round(float(crop_df["P"].mean()), 1),
                "K": round(float(crop_df["K"].mean()), 1),
                "temperature": round(float(crop_df["temperature"].mean()), 1),
                "humidity": round(float(crop_df["humidity"].mean()), 1),
                "ph": round(float(crop_df["ph"].mean()), 1),
                "rainfall": round(float(crop_df["rainfall"].mean()), 1)
            }
    return None

def parse_soil_inputs(message):
    msg = message.lower()
    patterns = {
        "N": [r"\bn\b\s*[:=]\s*([\d.]+)", r"nitrogen\s*[:=]?\s*([\d.]+)"],
        "P": [r"\bp\b\s*[:=]\s*([\d.]+)", r"phosphorus\s*[:=]?\s*([\d.]+)"],
        "K": [r"\bk\b\s*[:=]\s*([\d.]+)", r"potassium\s*[:=]?\s*([\d.]+)"],
        "ph": [r"\bph\b\s*[:=]\s*([\d.]+)", r"soil\s*ph\s*[:=]?\s*([\d.]+)"],
        "temperature": [r"\btemp\w*\b\s*[:=]\s*([\d.]+)", r"temperature\s*[:=]?\s*([\d.]+)"],
        "humidity": [r"\bhumid\w*\b\s*[:=]\s*([\d.]+)", r"humidity\s*[:=]?\s*([\d.]+)"],
        "rainfall": [r"\brain\w*\b\s*[:=]\s*([\d.]+)", r"rainfall\s*[:=]?\s*([\d.]+)", r"precip\w*\s*[:=]?\s*([\d.]+)"]
    }
    
    extracted = {}
    for key, regexes in patterns.items():
        for regex in regexes:
            match = re.search(regex, msg)
            if match:
                try:
                    extracted[key] = float(match.group(1))
                    break
                except ValueError:
                    pass
    return extracted

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500
    try:
        data = request.get_json(force=True)
        features = []
        for field in FEATURE_NAMES:
            if field not in data:
                return jsonify({"error": f"Missing field: '{field}'"}), 400
            try:
                features.append(float(data[field]))
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value for '{field}': must be a number"}), 400

        features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        top5 = sorted(zip(model.classes_, probabilities), key=lambda x: -x[1])[:5]
        importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
        confidence = round(float(max(probabilities)) * 100, 1)

        crop_key = prediction.lower()
        info = CROP_INFO.get(crop_key, {"emoji": "🌱", "season": "N/A", "water": "N/A", "tip": "No additional info available."})

        prediction_history.insert(0, {
            "crop": prediction.capitalize(),
            "confidence": confidence,
            "emoji": info["emoji"],
            "inputs": dict(zip(FEATURE_NAMES, features)),
            "time": datetime.now().strftime("%H:%M:%S")
        })
        if len(prediction_history) > 10:
            prediction_history.pop()

        return jsonify({
            "prediction": prediction.capitalize(),
            "confidence": confidence,
            "top5": [{"crop": c.capitalize(), "probability": round(float(p) * 100, 1)} for c, p in top5],
            "feature_importances": importances,
            "crop_info": info,
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/history", methods=["GET"])
def history():
    return jsonify(prediction_history)


@app.route("/dataset-stats", methods=["GET"])
def dataset_stats():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "Crop_recommendation.csv"))
        stats = df.describe().round(2).to_dict()
        crop_counts = df["label"].value_counts().to_dict()
        return jsonify({"stats": stats, "crop_counts": crop_counts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/weather", methods=["POST"])
def weather():
    data = request.get_json(force=True)
    location = data.get("location", "").strip()
    if not location:
        return jsonify({"error": "Please provide a location."}), 400
    

    try:
        weather_data = get_weather_for_location(location)
        return jsonify(weather_data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except requests.exceptions.HTTPError as e:
        error_msg = None
        try:
            payload = e.response.json() if e.response is not None else {}
            if isinstance(payload, dict):
                if "error" in payload:
                    err_val = payload["error"]
                    if isinstance(err_val, dict):
                        error_msg = err_val.get("message")
                    else:
                        error_msg = str(err_val)
                if not error_msg:
                    error_msg = payload.get("message")
        except Exception:
            pass
        if not error_msg:
            error_msg = str(e)
        return jsonify({"error": error_msg}), e.response.status_code if e.response is not None else 500
    except Exception as e:
        return jsonify({"error": f"Weather lookup failed: {str(e)}"}), 500


# ── Chatbot API Helpers & Endpoints ───────────────────────────────────────────

def get_gemini_response(api_key, conversation_history, user_message):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    system_text = (
        "You are AgriSmart AI, a helpful, intelligent agricultural advisor chatbot for farmers. "
        "Your goal is to provide fast, accurate, and friendly conversational responses to queries related to: "
        "crop recommendations, fertilizer suggestions, soil health (pH, N-P-K ratios), irrigation methods, "
        "weather impacts, pest control, plant diseases, organic farming, crop yield improvements, and seasonal advice. "
        "Format your response cleanly using bullet points, short paragraphs, or bold text for readability. "
        "Keep answers actionable and concise so they look good in a chat window. "
        "If you write equations or values, keep them simple and explain them in plain language. "
        "If the user asks something completely unrelated to agriculture, farming, crops, weather, or soil, "
        "gently guide them back to agricultural topics."
    )
    
    contents = []
    # Build contents from DB conversation history
    for msg in conversation_history:
        contents.append({"role": "user", "parts": [{"text": msg['user']}]})
        contents.append({"role": "model", "parts": [{"text": msg['bot']}]})
        
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    
    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": system_text}]
        }
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def get_openai_response(api_key, conversation_history, user_message):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    system_text = (
        "You are AgriSmart AI, a helpful, intelligent agricultural advisor chatbot for farmers. "
        "Your goal is to provide fast, accurate, and friendly conversational responses to queries related to: "
        "crop recommendations, fertilizer suggestions, soil health (pH, N-P-K ratios), irrigation methods, "
        "weather impacts, pest control, plant diseases, organic farming, crop yield improvements, and seasonal advice. "
        "Format your response cleanly using bullet points, short paragraphs, or bold text for readability. "
        "Keep answers actionable and concise so they look good in a chat window. "
        "If the user asks something completely unrelated to agriculture, farming, crops, weather, or soil, "
        "gently guide them back to agricultural topics."
    )
    
    messages = [{"role": "system", "content": system_text}]
    for msg in conversation_history:
        messages.append({"role": "user", "content": msg['user']})
        messages.append({"role": "assistant", "content": msg['bot']})
        
    messages.append({"role": "user", "content": user_message})
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def get_fallback_response(user_message):
    msg = user_message.lower()
    
    # 1. Parse soil values and run prediction if inputs are found
    soil_values = parse_soil_inputs(user_message)
    if soil_values and model is not None and scaler is not None:
        features = []
        feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        for f in feature_names:
            val = soil_values.get(f)
            if val is None:
                if f == "ph" and "ph" in soil_values:
                    val = soil_values["ph"]
                else:
                    val = DATASET_DEFAULTS.get(f, 50.0)
            features.append(val)
            
        try:
            features_df = pd.DataFrame([features], columns=feature_names)
            features_scaled = scaler.transform(features_df)
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = round(float(max(probabilities)) * 100, 1)
            
            crop_key = prediction.lower()
            info = CROP_INFO.get(crop_key, {"emoji": "🌱", "season": "N/A", "water": "N/A", "tip": "No additional info available."})
            emoji = info["emoji"]
            
            parsed_summary = []
            for f in feature_names:
                display_name = {
                    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
                    "temperature": "Temperature", "humidity": "Humidity", "ph": "pH", "rainfall": "Rainfall"
                }[f]
                if f in soil_values or (f == "ph" and "ph" in soil_values):
                    val = soil_values.get(f) if f in soil_values else soil_values.get("ph")
                    parsed_summary.append(f"• <strong>{display_name}:</strong> {val} (parsed)")
                else:
                    parsed_summary.append(f"• <strong>{display_name}:</strong> {round(DATASET_DEFAULTS[f], 1)} (estimate)")
                    
            return "<strong>Soil-Based Crop Recommendation:</strong><br>" \
                   "I detected the following parameters from your message:<br>" + "<br>".join(parsed_summary) + "<br><br>" \
                   f"Based on our machine learning Random Forest model, the recommended crop for these conditions is " \
                   f"<strong>{prediction.capitalize()}</strong> {emoji} with <strong>{confidence}%</strong> confidence!<br><br>" \
                   f"<em>Crop Tip:</em> {info['tip']} (Best in {info['season']} season, requires {info['water']} water)."
        except Exception as pred_err:
            print(f"Chatbot ML Prediction fallback failed: {pred_err}")

    # 2. Check for crop name queries
    detected_crop = None
    for crop in CROP_INFO.keys():
        if re.search(r'\b' + re.escape(crop) + r's?\b', msg):
            detected_crop = crop
            break
            
    if detected_crop:
        info = CROP_INFO[detected_crop]
        stats = get_crop_stats(detected_crop)
        
        if stats:
            stats_text = f"• <strong>Ideal N-P-K Ratio:</strong> N:{stats['N']} · P:{stats['P']} · K:{stats['K']}\n" \
                         f"• <strong>Ideal Soil pH:</strong> {stats['ph']}\n" \
                         f"• <strong>Ideal Climate:</strong> Temp: {stats['temperature']}°C · Humidity: {stats['humidity']}% · Rainfall: {stats['rainfall']}mm\n"
        else:
            stats_text = ""
            
        if "fertilizer" in msg or "nutrient" in msg or "npk" in msg:
            n_val = stats['N'] if stats else 'Moderate'
            p_val = stats['P'] if stats else 'Moderate'
            k_val = stats['K'] if stats else 'Moderate'
            return f"<strong>Fertilizer Advice for {detected_crop.capitalize()}:</strong> {info['emoji']}<br><br>" \
                   f"Based on our dataset averages, the ideal soil nutrients for {detected_crop} are:<br>" \
                   f"• <strong>Nitrogen (N):</strong> {n_val}<br>" \
                   f"• <strong>Phosphorus (P):</strong> {p_val}<br>" \
                   f"• <strong>Potassium (K):</strong> {k_val}<br><br>" \
                   f"<em>Tips for N-P-K management:</em><br>" \
                   f"• For low Nitrogen: Apply compost, well-rotted manure, or urea.<br>" \
                   f"• For low Phosphorus: Add bone meal or rock phosphate.<br>" \
                   f"• For low Potassium: Apply wood ash or organic potash.<br><br>" \
                   f"<strong>Farming Tip:</strong> {info['tip']}"
        else:
            return f"<strong>Crop Guide: {detected_crop.capitalize()}</strong> {info['emoji']}<br><br>" \
                   f"Here is what our agricultural database recommends for growing {detected_crop}:<br>" \
                   f"• <strong>Season:</strong> {info['season']}<br>" \
                   f"• <strong>Water Needs:</strong> {info['water']} ({'needs high water/irrigation' if info['water'].lower() == 'high' else 'moderate watering' if info['water'].lower() == 'medium' else 'drought-tolerant; low water'})<br>" \
                   f"{stats_text.replace(chr(10), '<br>')}" \
                   f"• <strong>Farming Tip:</strong> {info['tip']}"

    # 3. Check specific keywords
    if any(greet in msg for greet in ["hello", "hi", "hey", "greetings", "help", "who are you"]):
        return "Hello! I am your AgriSmart AI advisor. How can I help you today? 🌾<br><br>" \
               "You can ask me about:<br>" \
               "• <strong>fertilizer tips</strong> for crops (e.g. 'fertilizer tips for rice')<br>" \
               "• <strong>soil health</strong> (clay, sandy, loam advice)<br>" \
               "• <strong>organic farming & pest control</strong><br>" \
               "• <strong>irrigation & weather impacts</strong><br>" \
               "• <strong>detailed guide on 22 specific crops</strong> (e.g., banana, grapes, mango)<br><br>" \
               "<em>💡 Pro-tip:</em> Paste your Gemini API Key in the chat settings gear icon to enable full conversational AI mode!"

    elif "clay" in msg:
        return "<strong>Clay Soil Advice:</strong><br><br>" \
               "Clay soil has excellent water retention and nutrient content, but drains slowly and can compact easily.<br>" \
               "• <strong>Suitable Crops:</strong> Rice 🌾, Jute 🌾, and Cotton 🌿 (if drainage is managed).<br>" \
               "• <strong>Management:</strong> Avoid working clay soil when wet to prevent compaction. Mix in compost or organic matter to improve aeration."
               
    elif "sandy" in msg:
        return "<strong>Sandy Soil Advice:</strong><br><br>" \
               "Sandy soil is free-draining and warms up quickly, but has poor water and nutrient retention.<br>" \
               "• <strong>Suitable Crops:</strong> Watermelon 🍉, Muskmelon 🍈, Coconut 🥥, Chickpea 🫘, and Lentil 🫘.<br>" \
               "• <strong>Management:</strong> Use drip irrigation to prevent runoff. Apply organic mulch and compost regularly to build soil structure and retain moisture."
               
    elif "loam" in msg:
        return "<strong>Loam Soil Advice:</strong><br><br>" \
               "Loamy soil is the ideal soil type for farming, offering a balanced texture that retains both moisture and nutrients while draining well.<br>" \
               "• <strong>Suitable Crops:</strong> Maize 🌽, Banana 🍌, Mango 🥭, Coffee ☕, Cotton 🌿, Papaya 🍈, and other vegetables/grains.<br>" \
               "• <strong>Management:</strong> Maintain its high fertility by practicing crop rotation and adding light organic compost annually."

    elif "fertilizer" in msg or "nutrient" in msg or "potassium" in msg or "nitrogen" in msg or "phosphorus" in msg:
        return "<strong>Fertilizer Advice:</strong> Sizable yields require proper soil nutrient balance:<br>" \
               "• <em>Nitrogen (N):</em> For leafy growth. Add compost, manure, or urea.<br>" \
               "• <em>Phosphorus (P):</em> For root growth and flowers. Add bone meal or superphosphate.<br>" \
               "• <em>Potassium (K):</em> For disease resistance. Add potash or wood ash.<br><br>" \
               "Test your soil using our inputs page for specific crop advice! You can also type: 'N=90, P=40, K=40' to predict right here."
               
    elif "crop" in msg or "recommend" in msg or "grow" in msg:
        return "<strong>Crop Recommendation:</strong> AgriSmart AI uses a Random Forest model with 99.32% accuracy. " \
               "Please input your field parameters (N, P, K, pH, rainfall, temperature, humidity) in the " \
               "<strong>Predict</strong> tab above. The model will analyze them and recommend the most suitable crop!"
               
    elif "soil" in msg or "ph" in msg or "acid" in msg or "alkaline" in msg:
        return "<strong>Soil Health Tips:</strong><br>" \
               "• Most crops thrive in a slightly acidic to neutral pH (6.0 - 7.0).<br>" \
               "• If soil is too acidic (pH < 5.5), apply agricultural lime.<br>" \
               "• If soil is too alkaline (pH > 7.5), apply sulfur or organic compost.<br>" \
               "• Maintain soil organic matter by adding manure or mulch."
               
    elif "disease" in msg or "pest" in msg or "bug" in msg or "insect" in msg:
        return "<strong>Pest & Disease Control:</strong><br>" \
               "• <em>Organic Pest Control:</em> Use neem oil spray, garlic-chili spray, or release beneficial insects (like ladybugs).<br>" \
               "• <em>Fungal Diseases:</em> Avoid watering leaves directly; ensure proper spacing for air flow, and apply biological fungicides if needed.<br>" \
               "• <em>Crop Rotation:</em> Prevent pest buildup by rotating crops seasonal-to-season."
               
    elif "weather" in msg or "rain" in msg or "climate" in msg or "temperature" in msg:
        return "<strong>Weather Impact:</strong> Climate governs crop growth:<br>" \
               "• <em>Kharif (Monsoon):</em> Rice, Jute, Maize, Cotton. Needs high rainfall and warmth.<br>" \
               "• <em>Rabi (Winter):</em> Wheat, Chickpea, Lentil. Needs cooler temperatures and moderate water.<br>" \
               "• <em>Zaid (Summer):</em> Watermelon, Muskmelon. Sown in dry hot months with irrigation."
               
    elif "irrigation" in msg or "water" in msg:
        return "<strong>Irrigation Methods:</strong> Choose based on your soil and crop:<br>" \
               "• <em>Drip Irrigation:</em> Saves up to 50% water. Best for clay/loam soils, fruit crops, and vegetables.<br>" \
               "• <em>Sprinkler:</em> Mimics rain. Good for sandy soils and flat terrains.<br>" \
               "• <em>Furrow/Flood:</em> Traditional method. Used for rice and sugarcane, but has high water loss."
               
    elif "organic" in msg:
        return "<strong>Organic Farming Best Practices:</strong><br>" \
               "• Use compost, vermicompost, and green manure instead of chemical fertilizers.<br>" \
               "• Deploy bio-pesticides (neem oil) and practice crop rotation.<br>" \
               "• Cover cropping helps retain soil moisture and prevents weed growth."
               
    elif "yield" in msg or "improve" in msg or "increase" in msg:
        return "<strong>Improving Crop Yields:</strong><br>" \
               "• Choose high-yielding, certified seed varieties suited for your region.<br>" \
               "• Practice crop rotation to maintain soil microbial health.<br>" \
               "• Adopt efficient irrigation systems like drip watering.<br>" \
               "• Monitor soil health regularly using our AgriSmart AI input values."
               
# duplicate else removed
               
    else:
        return "I am running in offline mode. Please configure `GEMINI_API_KEY` or `OPENAI_API_KEY` in your `.env` file to unlock dynamic conversations. " \
               "Currently, I can answer questions about: <strong>fertilizer</strong>, <strong>crop selection</strong>, " \
               "<strong>soil health</strong>, <strong>diseases</strong>, <strong>weather</strong>, <strong>irrigation</strong>, " \
               "<strong>organic farming</strong>, and <strong>improving yield</strong>."


@app.route("/chat", methods=["POST"])
@app.route("/get-response", methods=["POST"])
def chat_endpoint():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400
    
    user_id = data.get("user_id", "").strip()
    message = data.get("message", "").strip()
    
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400
    if not message:
        return jsonify({"error": "Missing message parameter"}), 400
        
    if is_rate_limited(user_id):
        return jsonify({"error": "Too many requests. Please wait a moment before sending another message."}), 429
        
    # Retrieve previous conversation context (last 10 interactions)
    history = get_chat_history_from_db(user_id, limit=10)
    
    # Reload OS environment keys in case .env was edited
    load_dotenv_from_file(os.path.join(BASE_DIR, ".env"))
    
    # Read API Keys (fresh from client payload and OS environment)
    gemini_key = data.get("gemini_key", "").strip() or os.environ.get("GEMINI_API_KEY")
    openai_key = data.get("openai_key", "").strip() or os.environ.get("OPENAI_API_KEY")
    
    try:
        if gemini_key:
            response = get_gemini_response(gemini_key, history, message)
        elif openai_key:
            response = get_openai_response(openai_key, history, message)
        else:
            response = get_fallback_response(message)
    except Exception as e:
        print(f"AI Chatbot Error: {e}")
        response = f"I encountered an error trying to process that with the AI models. Here is some general information: <br><br>{get_fallback_response(message)}"
        
    # Save the interaction to our SQLite database
    try:
        save_chat_to_db(user_id, message, response)
    except Exception as db_err:
        print(f"Database Save Error: {db_err}")
        
    return jsonify({
        "response": response,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route("/chat-history", methods=["GET"])
def chat_history_endpoint():
    user_id = request.args.get("user_id", "").strip()
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400
        
    try:
        history = get_chat_history_from_db(user_id, limit=50)
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve chat history: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)