import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # bu dosyanın olduğu klasör = services
CSV_DIR = os.path.join(BASE_DIR,"services")

# CSV’leri oku
no2 = pd.read_csv(os.path.join(CSV_DIR,"state_avg_no2.csv"))
o2o2 = pd.read_csv(os.path.join(CSV_DIR,"state_avg_o2o2.csv"))
o3 = pd.read_csv(os.path.join(CSV_DIR,"state_avg_o3.csv"))
hcho = pd.read_csv(os.path.join(CSV_DIR,"state_avg_hcho.csv"))

df = no2.copy()

# Ortak kolonlara göre birleştir
df = df.merge(o3, on=["state_name","date"])
df = df.merge(hcho, on=["state_name","date"])
df = df.merge(o2o2, on=["state_name","date"])
'''
df['activity_ok'] = df['no2'] + df['avg_o3'] + df['avg_hcho']
df['activity_ok'] = df['activity_ok'].apply(lambda x: 1 if x < 25 else 0)
'''
# Tek CSV olarak kaydet
OUTPUT_PATH = os.path.join(BASE_DIR, "training_data.csv")
df.to_csv(OUTPUT_PATH, index=False)
print(f"Training data kaydedildi: {OUTPUT_PATH}")