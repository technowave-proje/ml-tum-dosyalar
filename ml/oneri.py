# ml/advisor.py
from typing import Dict

# ----------------------
# Aktivite Önerileri
# ----------------------
def check_activity(activity, temp, wind, rain, aqi, pm25) -> Dict:
    rain = bool(rain)
    status = "Uygun"
    reasons = []

    if aqi >= 4 or pm25 > 50:
        status = "Uygun Değil"
        reasons.append("Hava kalitesi yüksek (AQI/PM2.5)")

    if activity == "yüzme":
        if temp < 20:
            status = "Uygun Değil"
            reasons.append("Sicaklik yüzme için düsük")
        if rain:
            status = "Uygun Değil"
            reasons.append("Yagmur yüzme için uygun degil")
        if temp >= 20 and not rain:
            reasons.append("Hava yüzme için uygun")
    
    elif activity == "koşu":
        if temp < 5 or temp > 32:
            status = "Uygun Değil"
            reasons.append("Sicaklik kosu için uygun degil")
        if wind > 8:
            status = "Uygun Değil"
            reasons.append("Rüzgar kosu için yuksek")
        if 5 <= temp <= 32 and wind <= 8:
            reasons.append("Hava kosu için uygun")
    
    # Diğer aktiviteler aynı şekilde devam eder...
    elif activity == "golf":
        if wind > 10:
            status = "Uygun Değil"
            reasons.append("Rüzgar golf için uygun değil")
        if rain:
            status = "Uygun Değil"
            reasons.append("Yağmur golf için uygun değil")
        if wind <= 10 and not rain:
            reasons.append("Hava golf için uygun")
    
    elif activity in ["yürüyüş", "doğa yürüyüşü"]:
        if temp < 5 or temp > 35:
            status = "Uygun Değil"
            reasons.append("Sicaklik yürüyüş için uygun değil")
        if rain:
            status = "Uygun Değil"
            reasons.append("Yağmur yürüyüş için uygun değil")
        if 5 <= temp <= 35 and not rain:
            reasons.append("Hava yürüyüş için uygun")
    
    elif activity == "bisiklet":
        if wind > 9:
            status = "Uygun Değil"
            reasons.append("Rüzgar bisiklet için tehlikeli")
        if rain:
            status = "Uygun Değil"
            reasons.append("Yağmur bisiklet için uygun değil")
        if wind <= 9 and not rain:
            reasons.append("Hava bisiklet için uygun")
    
    elif activity == "tenis":
        if wind > 12:
            status = "Uygun Değil"
            reasons.append("Rüzgar tenis için çok şiddetli")
        if rain:
            status = "Uygun Değil"
            reasons.append("Yağmur tenis için uygun değil")
        if wind <= 12 and not rain:
            reasons.append("Hava tenis için uygun")
    
    elif activity == "açik hava sporu":
        if temp < 0 or temp > 36:
            status = "Uygun Değil"
            reasons.append("Sicaklik açik hava sporu için uygun değil")
        if rain:
            status = "Uygun Değil"
            reasons.append("Yağmur açik hava sporlari için uygun değil")
        if wind > 15:
            status = "Uygun Değil"
            reasons.append("Rüzgar açik hava sporlari için tehlikeli")
        if 0 <= temp <= 36 and not rain and wind <= 15:
            reasons.append("Hava açik hava sporlari için uygun")

    return {"aktivite": activity, "durum": status, "neden": reasons}


# ----------------------
# Sağlık Önerileri
# ----------------------
def age_group_advice(age_group, temp, aqi, pm25):
    status = "Uygun"
    reasons = []

    if age_group == "çocuk":
        if aqi >= 3 or pm25 > 40:
            status = "Uygun Değil"
            reasons.append("Çocuklar için hava kalitesi riskli")
        if temp < 10 or temp > 30:
            status = "Uygun Değil"
            reasons.append("Çocuklar için sicaklik uygun değil")
        if status == "Uygun":
            reasons.append("Çocuklar için hava uygun")
    
    elif age_group == "yetişkin":
        if aqi >= 4 or pm25 > 50:
            status = "Uygun Değil"
            reasons.append("Yetişkinler için hava kalitesi riskli")
        if temp < 5 or temp > 32:
            status = "Uygun Değil"
            reasons.append("Yetişkinler için sicaklik uygun değil")
        if status == "Uygun":
            reasons.append("Yetişkinler için hava uygun")
    
    elif age_group == "yaşli":
        if aqi >= 3 or pm25 > 40:
            status = "Uygun Değil"
            reasons.append("Yaşlılar için hava kalitesi riskli")
        if temp < 10 or temp > 28:
            status = "Uygun Değil"
            reasons.append("Yaşlılar için sıcaklık uygun değil")
        if status == "Uygun":
            reasons.append("Yaşlılar için hava uygun")
    
    return {"age_group": age_group, "durum": status, "neden": "; ".join(reasons)}

def pregnancy_advice(status_group, temp, aqi, pm25):
    status = "Uygun"
    reasons = []

    if status_group == "hamile":
        if aqi >= 3 or pm25 > 40:
            status = "Uygun Değil"
            reasons.append("Hamileler için hava kalitesi riskli")
        if temp < 10 or temp > 32:
            status = "Uygun Değil"
            reasons.append("Hamileler için sicaklik uygun değil")
        if status == "Uygun":
            reasons.append("Hamileler için hava uygun")
    
    elif status_group == "emziren":
        if aqi >= 4 or pm25 > 50:
            status = "Uygun Değil"
            reasons.append("Emziren anneler için hava kalitesi riskli")
        if temp < 10 or temp > 32:
            status = "Uygun Değil"
            reasons.append("Emziren anneler için sicaklik uygun değil")
        if status == "Uygun":
            reasons.append("Emziren anneler için hava uygun")
    
    return {"status_group": status_group, "durum": status, "neden": "; ".join(reasons)}

def respiratory_advice(disease, aqi, pm25):
    status = "Uygun"
    reasons = []

    if disease in ["astım", "KOAH"]:
        if pm25 > 30 or aqi >= 3:
            status = "Uygun Değil"
            reasons.append(f"{disease} hastaları için hava kalitesi riskli")
        else:
            reasons.append(f"{disease} hastaları için hava uygun")
    
    elif disease == "sinüzit":
        if aqi >= 4:
            status = "Uygun Değil"
            reasons.append("Sinüzit hastaları için hava kalitesi yüksek")
        else:
            reasons.append("Sinüzit hastaları için hava uygun")
    
    elif disease == "kronik bronşit":
        if pm25 > 25 or aqi >= 3:
            status = "Uygun Değil"
            reasons.append("Kronik bronşit hastaları için hava kalitesi riskli")
        else:
            reasons.append("Kronik bronşit hastaları için hava uygun")
    
    elif disease == "alerji":
        if aqi >= 3:
            status = "Uygun Değil"
            reasons.append("Alerjisi olanlar için hava kalitesi riskli")
        else:
            reasons.append("Alerjisi olanlar için hava uygun")
    
    return {"disease": disease, "durum": status, "neden": "; ".join(reasons)}

def cardio_metabolic_advice(disease, temp, aqi, pm25):
    status = "Uygun"
    reasons = []

    if disease == "diyabet":
        if temp < 5 or temp > 32 or aqi >= 4 or pm25 > 50:
            status = "Uygun Değil"
            reasons.append("Diyabet hastaları için riskli hava koşulları")
        else:
            reasons.append("Diyabet hastaları için hava uygun")
    
    elif disease == "kalp hastalığı":
        if temp < 5 or temp > 30 or aqi >= 4 or pm25 > 50:
            status = "Uygun Değil"
            reasons.append("Kalp hastaları için riskli hava koşulları")
        else:
            reasons.append("Kalp hastaları için hava uygun")
    
    elif disease == "hipertansiyon":
        if temp < 5 or temp > 30 or aqi >= 4 or pm25 > 50:
            status = "Uygun Değil"
            reasons.append("Hipertansiyon hastaları için riskli hava koşulları")
        else:
            reasons.append("Hipertansiyon hastaları için hava uygun")
    
    return {"disease": disease, "durum": status, "neden": "; ".join(reasons)}
