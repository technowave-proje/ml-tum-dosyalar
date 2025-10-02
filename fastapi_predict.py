''' 
from fastapi import APIRouter , HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# -----------------------------
# FastAPI uygulaması
# -----------------------------
router=APIRouter(
    prefix="/predict"
)

# -----------------------------
# Kullanıcı girdisi (veri modeli)
# -----------------------------
class UserInput(BaseModel):
    # 🌤 OpenWeather (Weather API)
    temperature: float
    feels_like: float
    pressure: int
    humidity: float
    wind_speed: float
    rain: bool
    
    # 🌍 OpenWeather AirPollution
    aqi: int          # 1-5 arası
    pm25: float
    pm10: float
    so2: float
    no2: float
    co: float
    o3: float
    
    # 🚀 NASA TEMPO L3
    o2_o2: float
    o3_tempo: float
    no2_tempo: float
    hcho: float

# -----------------------------
# Model yükleme
# -----------------------------
try:
    model = joblib.load("aqi_model.pkl")
except:
    model = None
    print("⚠ Model dosyası bulunamadı: aqi_model.pkl")

# -----------------------------
# Tahmin endpointi
# -----------------------------
@router.post("/predict")
async def predict(data: UserInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model yüklenemedi!")

    try:
        # Verileri numpy array'e çevir
        features = np.array([[ 
            data.temperature,
            data.feels_like,
            data.pressure,
            data.humidity,
            data.wind_speed,
            1 if data.rain else 0,   # boolean -> int
            
            data.aqi,
            data.pm25,
            data.pm10,
            data.so2,
            data.no2,
            data.co,
            data.o3,
            
            data.o2_o2,
            data.o3_tempo,
            data.no2_tempo,
            data.hcho
        ]])

        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Çalıştırma (geliştirme için)
# -----------------------------
if "name" == "main":
    uvicorn.run(app, host="0.0.0.0",port=8000) 
    '''


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
from ml.predictor import make_prediction  # ✅ predictor.py’den import et

router = APIRouter(prefix="/predict")

class UserInput(BaseModel):
    temperature: float
    feels_like: float
    pressure: int
    humidity: float
    wind_speed: float
    rain: bool
    aqi: int
    pm25: float
    pm10: float
    so2: float
    no2: float
    co: float
    o3: float
    o2_o2: float
    o3_tempo: float
    no2_tempo: float
    hcho: float

# Model yükle
try:
    model = joblib.load("aqi_model.pkl")
except:
    model = None
    print("⚠ Model dosyası bulunamadı: aqi_model.pkl")

@router.post("/predict")
async def predict(data: UserInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model yüklenemedi!")

    # ✅ Dict oluştur
    input_dict = {
        "state_name": "default_state",  # sabit veya kullanıcıdan al
        "date": "2025-10-02",           # sabit veya kullanıcıdan al
        "no2": data.no2,
        "avg_o3": data.o3,
        "avg_hcho": data.hcho,
        "avg_o2o2": data.o2_o2,
        "age_group": "adult",
        "pregnancy_status": "none",
        "respiratory_disease": "no",
        "cardio_disease": "no"
    }

    try:
        result = make_prediction(model, input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
