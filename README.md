# Health Monitoring Using Breath Pattern Analysis

An AI-assisted respiratory health monitoring application powered by a **scikit-learn Random Forest model** (`breathing_random_forest.pkl`) and a dual-service architecture with **Next.js App Router** and a **Python FastAPI ML Microservice**.

---

## 🏗️ Architecture Overview

```
Frontend (React UI)
   │
   ▼
Next.js API Gateway (POST /api/predict)
   │
   ▼ (multipart/form-data audio forward)
Python FastAPI ML Service (http://127.0.0.1:8000/predict)
   │
   ├── Audio Preprocessing (48kHz Mono)
   ├── 4th-order Butterworth Bandpass Filter (80Hz - 8000Hz)
   ├── Mel Spectrogram (128 mels, n_fft=2048, hop=512) -> dB Conversion
   ├── Target Shape (128 × 512) with -80 dB Padding & [0, 1] Normalization
   ├── 512 Statistical Features [Mean (128), Std (128), Max (128), Min (128)]
   └── Random Forest (.pkl) Inference
   │
   ▼
JSON Prediction Result -> Next.js API -> Frontend Display
```

---

## ⚙️ Requirements & Installation

### 1. Prerequisites
- **Node.js**: `v18.x` or higher (`npm` package manager)
- **Python**: `v3.10` or `v3.11` (`pip` package manager)

### 2. Model Location
Ensure the trained model binary `breathing_random_forest.pkl` is located at:
```
ml-service/models/breathing_random_forest.pkl
```

---

## 🚀 Step-by-Step Setup & Running

### Terminal 1: Start Python FastAPI ML Service (Port 8000)

```bash
# Navigate to ML service directory
cd ml-service

# (Optional but recommended) Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install Python ML dependencies
pip install -r requirements.txt

# Start FastAPI Uvicorn Server on port 8000
python3 -m uvicorn main:app --reload --port 8000
```
- **ML Service Health Check**: `http://127.0.0.1:8000/health`
- **ML Swagger API Docs**: `http://127.0.0.1:8000/docs`

---

### Terminal 2: Start Next.js Frontend & API Gateway (Port 3000)

```bash
# In the project root directory
npm install

# Start Next.js Development Server
npm run dev
```
- **Application URL**: `http://localhost:3000`
- **API Gateway Endpoint**: `http://localhost:3000/api/predict`

---

## 🧪 Testing the API

### 1. Health Check Endpoints
```bash
# FastAPI Health Check
curl -s http://127.0.0.1:8000/health
# Expected: {"status":"ok","model":"breathing_random_forest"}

# Next.js API Gateway Health Check
curl -s http://localhost:3000/api/health
```

### 2. Test Audio Prediction Endpoint (`POST /api/predict`)
You can test the full pipeline with a sample `.wav` file:
```bash
curl -X POST http://localhost:3000/api/predict \
  -F "audio=@path/to/your/audio.wav"
```

#### Example Output Response:
```json
{
  "success": true,
  "prediction": 1,
  "label": "Positive-associated",
  "probability": 0.6186,
  "percentage": 61.86,
  "spectrogram_shape": [128, 512],
  "message": "The audio pattern is associated with the positive class in the dataset.",
  "disclaimer": "This is a research prototype and not a medical diagnostic system."
}
```

---

## 📂 Project Directory Structure

```
.
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   ├── predict/
│   │   │   │   └── route.js     # Next.js API Gateway (Forwards audio to FastAPI)
│   │   │   └── health/
│   │   │       └── route.js     # Health check status route
│   │   ├── layout.js
│   │   └── page.js
│   ├── components/              # Preserved React UI components
│   └── services/
│       └── predictionService.js # Frontend API service layer
├── ml-service/                  # Python FastAPI ML Microservice
│   ├── main.py                  # FastAPI application routes (/health, /predict)
│   ├── inference.py             # Exact 8-step ML preprocessing pipeline
│   ├── requirements.txt         # Python dependencies (librosa, scipy, sklearn, etc.)
│   └── models/
│       └── breathing_random_forest.pkl # Random Forest ML model (512 features)
├── .env.local                   # Environment variables (ML_SERVICE_URL)
├── package.json
└── README.md
```

---

## 🛠️ Troubleshooting

1. **`ML Prediction Service is currently unreachable`**:
   - Make sure FastAPI is running on `http://127.0.0.1:8000` in Terminal 1.
   - Verify `.env.local` contains `ML_SERVICE_URL=http://127.0.0.1:8000`.

2. **`Model file not found at ml-service/models/breathing_random_forest.pkl`**:
   - Verify `breathing_random_forest.pkl` is placed inside `ml-service/models/`.

3. **`Librosa audio load error / FFmpeg missing`**:
   - `librosa` natively processes `.wav` audio. If recording `.webm` audio in browsers, ensure `soundfile` or `audioread` Python libraries are installed (included in `requirements.txt`).
