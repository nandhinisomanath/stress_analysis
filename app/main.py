from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Load saved artifacts
model = joblib.load('../notebooks/stress_classifier.pkl')
scaler = joblib.load('../notebooks/scaler.pkl')
le_target = joblib.load('../notebooks/le_target.pkl')
le_gender = joblib.load('../notebooks/le_gender.pkl')
le_occupation = joblib.load('../notebooks/le_occupation.pkl')

app = FastAPI(title="Stress Level Prediction API")


class StressInput(BaseModel):
    age:                              int = Field(..., example=25)
    gender:                           str = Field(..., example='Female')
    occupation:                       str = Field(..., example='Student')
    daily_screen_time_hours:          float = Field(..., example=9.0)
    phone_usage_before_sleep_minutes: int = Field(..., example=90)
    sleep_duration_hours:             float = Field(..., example=5.0)
    sleep_quality_score:              float = Field(..., example=3.5)
    caffeine_intake_cups:             int = Field(..., example=4)
    physical_activity_minutes:        int = Field(..., example=10)
    notifications_received_per_day:   int = Field(..., example=180)
    mental_fatigue_score:             float = Field(..., example=8.5)


@app.post("/predict")
def predict(data: StressInput):
    row = {'age': data.age,
           'gender': le_gender.transform([data.gender])[0],
           'occupation': le_occupation.transform([data.occupation])[0],
           'daily_screen_time_hours': data.daily_screen_time_hours,
           'phone_usage_before_sleep_minutes': data.phone_usage_before_sleep_minutes,
           'sleep_duration_hours': data.sleep_duration_hours,
           'sleep_quality_score': data.sleep_quality_score,
           'caffeine_intake_cups': data.caffeine_intake_cups,
           'physical_activity_minutes': data.physical_activity_minutes,
           'notifications_received_per_day': data.notifications_received_per_day,
           'mental_fatigue_score': data.mental_fatigue_score,
           # Engineered features
           'screen_to_sleep_ratio': data.daily_screen_time_hours / data.sleep_duration_hours + 0.1,
           'pre_sleep_screen_burden': data.phone_usage_before_sleep_minutes * data.daily_screen_time_hours,
           'fatigue_sleep_interaction': data.mental_fatigue_score * (10 - data.sleep_quality_score),
           'activity_caffeine_ratio': data.physical_activity_minutes / (data.caffeine_intake_cups + 1)}

    df_row = pd.DataFrame([row])
    df_sc = scaler.transform(df_row)
    pred = model.predict(df_sc)
    proba = model.predict_proba(df_sc)[0]
    label = le_target.inverse_transform(pred)[0]
    conf = {le_target.classes_[i]
        : f'{p*100:.1f}%' for i, p in enumerate(proba)}

    return {'predicted_stress_level': label,
            'confidence': conf}


@app.get('/health')
def health():
    return {'status': 'ok',
            'classes': list(le_target.classes_)}
