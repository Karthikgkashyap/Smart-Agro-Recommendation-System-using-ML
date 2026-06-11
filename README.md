# AgriSmart AI — Crop Recommendation System

A Flask web app that recommends the best crop based on soil and climate parameters using a **Random Forest** model (99.32% accuracy).

---

## 📁 Project Structure

```
project/
├── app.py                  ← Flask backend (main entry point)
├── requirements.txt        ← Python dependencies
├── data/
│   └── Crop_recommendation.csv
├── models/
│   ├── rf_model.pkl        ← Trained Random Forest model
│   └── scaler.pkl          ← Fitted StandardScaler
├── src/
│   └── train_model.py      ← Re-train the model (optional)
├── static/
│   ├── css/style.css
│   └── js/main.js
└── templates/
    └── index.html
```

---

## 🚀 How to Run in VS Code

### Step 1 — Open the project folder in VS Code
```
File → Open Folder → select the `project` folder
```

### Step 2 — Open the integrated terminal
```
Terminal → New Terminal  (or Ctrl + `)
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
python app.py
```

### Step 5 — Open in browser
```
http://127.0.0.1:5000
```

---

## 🔄 Re-training the Model (optional)

The trained model files are already included in `models/`. If you want to retrain:

```bash
python src/train_model.py
```

---

## 🌾 Input Parameters

| Parameter   | Description              | Example  |
|-------------|--------------------------|----------|
| N           | Nitrogen ratio in soil   | 90       |
| P           | Phosphorus ratio         | 42       |
| K           | Potassium ratio          | 43       |
| Temperature | In Celsius               | 20.87    |
| Humidity    | Relative humidity %      | 82.0     |
| pH          | Soil pH value (0–14)     | 6.5      |
| Rainfall    | In mm                    | 202.9    |

---

## 🌦️ Weather API Integration

The app can auto-fill `temperature`, `humidity`, and `rainfall` from OpenWeatherMap.

1. Sign up at https://openweathermap.org/ and get an API key.
2. Set `OPENWEATHER_API_KEY` in your environment or add it to a `.env` file in the project root.

```bash
export OPENWEATHER_API_KEY="your_api_key_here"
```

On Windows PowerShell:

```powershell
$env:OPENWEATHER_API_KEY="your_api_key_here"
```

Or create `.env` in the project root:

```env
OPENWEATHER_API_KEY=your_api_key_here
```

Enter a location in the form, click **Use Weather Data**, then run Predict.

---

## ⚙️ API Endpoint

**POST** `/predict`

```json
Request:  { "N": "90", "P": "42", "K": "43", "temperature": "20.87", "humidity": "82.0", "ph": "6.5", "rainfall": "202.9" }
Response: { "prediction": "Rice" }
```
