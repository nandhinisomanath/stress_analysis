# 😴 Sleep, Screen Time & Stress Analysis
### Machine Learning Classification Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-orange?style=flat-square&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green?style=flat-square&logo=fastapi)
![Dataset](https://img.shields.io/badge/Dataset-15%2C000%20rows-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Project Overview

This project builds a **multiclass classification model** to predict a person's **stress level (Low / Medium / High)** based on their sleep patterns, mobile screen time habits, and daily lifestyle data.

The dataset contains **15,000 records** with **13 features** including sleep duration, sleep quality, daily screen time, phone usage before sleep, mental fatigue, caffeine intake, and physical activity.

---

## 🎯 Problem Statement

> **"Can we predict a person's stress level from their daily sleep and screen time behaviour?"**

- **Target Variable** : `stress_level` (continuous 1–10) → binned into **Low / Medium / High**
- **Task Type** : Multiclass Classification
- **Dataset Size** : 15,000 rows × 13 columns
- **Source** : [Kaggle — Sleep, Screen Time & Stress Dataset](https://www.kaggle.com/)

---

## 📁 Project Structure

```
stress-ml-project/
│
├── sleep_mobile_stress_dataset_15000.csv   # Raw dataset
├── stress_ml_project.py                    # Main ML pipeline (all phases)
├── deployment_app.py                       # FastAPI deployment app
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
│
├── outputs/
│   ├── stress_classifier.pkl               # Trained model
│   ├── scaler.pkl                          # StandardScaler
│   ├── le_target.pkl                       # Target label encoder
│   ├── le_gender.pkl                       # Gender label encoder
│   └── le_occupation.pkl                   # Occupation label encoder
│
└── plots/
    ├── eda_plots.png                        # EDA visualizations
    ├── confusion_matrix.png                 # Confusion matrix
    ├── feature_importance.png               # Feature importances
    └── roc_curves.png                       # ROC-AUC curves
```

---

## 📊 Dataset Description

| Column | Type | Range | Description |
|---|---|---|---|
| `user_id` | Integer | 1–15,000 | Row identifier (dropped) |
| `age` | Integer | 18–59 | Age of the person |
| `gender` | String | Male/Female/Other | Gender |
| `occupation` | String | 8 categories | Job role |
| `daily_screen_time_hours` | Float | 1–10 | Daily screen usage (hrs) |
| `phone_usage_before_sleep_minutes` | Integer | 0–300 | Phone use before bed (mins) |
| `sleep_duration_hours` | Float | 4–9 | Hours of sleep |
| `sleep_quality_score` | Float | 1–10 | Self-rated sleep quality |
| `stress_level` | Float | 1–10 | **Target** — binned into 3 classes |
| `caffeine_intake_cups` | Integer | 0–10 | Daily caffeine intake |
| `physical_activity_minutes` | Integer | 0–300 | Exercise per day (mins) |
| `notifications_received_per_day` | Integer | 0–500 | Daily phone notifications |
| `mental_fatigue_score` | Float | 1–10 | Mental tiredness score |

### Target Class Distribution

| Class | Stress Range | Count | Percentage |
|---|---|---|---|
| Low | 1 – 4 | 2,734 | 18.2% |
| Medium | 4 – 7 | 4,189 | 27.9% |
| High | 7 – 10 | 8,077 | 53.8% |

---

## 🔍 Key EDA Findings

- **`mental_fatigue_score`** has the strongest correlation with stress → **+0.95**
- **`daily_screen_time_hours`** is the second strongest predictor → **+0.88**
- **`sleep_quality_score`** is the strongest negative predictor → **−0.86**
- **`sleep_duration_hours`** has almost zero correlation → **−0.005** (quality matters more than quantity)
- High-stress people average **7.4 hrs** screen time vs **2.25 hrs** for Low-stress people
- All 8 occupations are evenly distributed (~1,800–1,960 rows each)

---

## ⚙️ Project Phases

| Phase | Description |
|---|---|
| **Phase 1** | Problem Setup & Data Loading |
| **Phase 2** | Exploratory Data Analysis (EDA) |
| **Phase 3** | Data Preprocessing & Feature Engineering |
| **Phase 4** | Model Building & Comparison |
| **Phase 5** | Evaluation Metrics |
| **Phase 6** | Hyperparameter Tuning |
| **Phase 7** | Save Model Artifacts |
| **Phase 8** | Deployment with FastAPI |

---

## 🛠️ Feature Engineering

4 new features were created from existing columns:

```python
# Screen dominance over sleep
screen_to_sleep_ratio = daily_screen_time_hours / (sleep_duration_hours + 0.1)

# Combined phone-before-sleep pressure
pre_sleep_screen_burden = phone_usage_before_sleep_minutes * daily_screen_time_hours

# Fatigue amplified by poor sleep quality
fatigue_sleep_interaction = mental_fatigue_score * (10 - sleep_quality_score)

# Whether exercise offsets stimulant use
activity_caffeine_ratio = physical_activity_minutes / (caffeine_intake_cups + 1)
```

---

## 🤖 Models Trained

| Model | Validation Accuracy | CV F1 (Weighted) | Selected |
|---|---|---|---|
| Logistic Regression | 90.22% | 0.8971 | |
| **Random Forest** | **89.56%** | **0.8979** | **✓ Best** |
| Gradient Boosting | 89.29% | 0.8963 | |

> Model selected based on **5-fold cross-validated F1 score** (more reliable than raw accuracy)

---

## 📈 Final Model Performance

**Model : Random Forest Classifier (Tuned)**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| High | 0.91 | 0.93 | 0.92 | ~1,213 |
| Low | 0.87 | 0.83 | 0.85 | ~410 |
| Medium | 0.84 | 0.81 | 0.82 | ~627 |
| **Weighted Avg** | **0.89** | **0.89** | **0.888** | **2,250** |

```
Test Accuracy   : 88.76%
Weighted F1     : 0.888
Weighted AUC-ROC: 0.978  ← Excellent
```

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/nandhinisomanath/stress_analysis.git
cd stress_analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```


### 3. Start the API server

```bash
uvicorn app/main:app --reload --port 8000
```

### 4. Open Swagger UI

```
http://localhost:8000/docs
```

---

## 🌐 API Usage

### Predict Stress Level

**Endpoint** : `POST /predict`

**Request Body :**
```json
{
  "age": 22,
  "gender": "Female",
  "occupation": "Student",
  "daily_screen_time_hours": 9.0,
  "phone_usage_before_sleep_minutes": 90,
  "sleep_duration_hours": 5.0,
  "sleep_quality_score": 3.5,
  "caffeine_intake_cups": 4,
  "physical_activity_minutes": 10,
  "notifications_received_per_day": 180,
  "mental_fatigue_score": 8.5
}
```

**Response :**
```json
{
  "predicted_stress_level": "High",
  "confidence": {
    "High": "91.2%",
    "Medium": "6.4%",
    "Low": "2.4%"
  }
}
```

### Health Check

**Endpoint** : `GET /health`

```json
{
  "status": "ok",
  "model": "Random Forest",
  "classes": ["High", "Low", "Medium"]
}
```

---

## 📦 Requirements

```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.4.2
joblib==1.4.2
fastapi==0.111.0
uvicorn==0.29.0
pydantic==2.7.1
```

---

## 🧠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualization |
| Scikit-Learn | ML models, preprocessing, evaluation |
| Joblib | Model serialization |
| FastAPI | REST API deployment |
| Uvicorn | ASGI server |

---

## 📌 How to Contribute

1. Fork the repository
2. Create a new branch → `git checkout -b feature/your-feature`
3. Commit your changes → `git commit -m "Add your feature"`
4. Push to branch → `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub : [@nandhinisomanath](https://github.com/nandhinisomanath)
- Email : nandhinisomanath03@gmail.com

---

> ⭐ If this project helped you, please give it a star on GitHub!