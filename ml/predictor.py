# predictor.py
'''
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mysql.connector
from mysql.connector import Error
import os

# -----------------------------
# 1️⃣ Model Eğitimi CSV'den
# -----------------------------
def train_model_from_csv(csv_path=None, model_path="aqi_model.pkl"):
    """
    CSV'den veri alıp modeli eğitir ve kaydeder.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "training_data.csv")

    
    data = pd.read_csv(csv_path)
    print("CSV yüklendi, kolonlar:", data.columns.tolist())
    
    if "activity_ok" not in data.columns:
        raise KeyError("❌ CSV'de 'activity_ok' kolonu bulunamadı.")
    
    X = data.drop("activity_ok", axis=1)
    y = data["activity_ok"]
    
    # Kategorik kolonlar için one-hot encode
    categorical_cols = ["state_name", "date"]
    for col in categorical_cols:
        if col not in X.columns:
            X[col] = "none"
    X = pd.get_dummies(X, columns=categorical_cols)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_path)
    print(f"✅ Model CSV'den başarıyla eğitildi ve kaydedildi: {model_path}")
    return model

# -----------------------------
# 2️⃣ MySQL'den veri çekme
# -----------------------------
def load_data_from_mysql(host, user, password, database, table):
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        query = f"SELECT * FROM {table};"
        data = pd.read_sql(query, con=connection)
        connection.close()
        print(f"✅ MySQL'den veri çekildi: {table}")
        return data
    except Error as e:
        print(f"❌ MySQL bağlantı hatası: {e}")
        return None

# -----------------------------
# 3️⃣ MySQL verisi ile model eğitimi
# -----------------------------
def train_model_from_mysql(host, user, password, database, table, model_path="aqi_model.pkl"):
    data = load_data_from_mysql(host, user, password, database, table)
    if data is None or data.empty:
        print("❌ MySQL verisi boş veya çekilemedi.")
        return None
    
    if "activity_ok" not in data.columns:
        raise KeyError("❌ MySQL verisinde 'activity_ok' kolonu bulunamadı.")
    
    X = data.drop("activity_ok", axis=1)
    y = data["activity_ok"]
    
    categorical_cols = ["state_name", "date", "no2", "avg_o3", "avg_hcho", "avg_o2o2","activity_ok"]
    for col in categorical_cols:
        if col not in X.columns:
            X[col] = "none"
    X = pd.get_dummies(X, columns=categorical_cols)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, model_path)
    print(f"✅ Model MySQL verisi ile eğitildi ve kaydedildi: {model_path}")
    return model

# -----------------------------
# 4️⃣ Modeli yükleme
# -----------------------------
def load_model(model_path="aqi_model.pkl"):
    try:
        model = joblib.load(model_path)
        print("📂 Model başarıyla yüklendi.")
        return model
    except FileNotFoundError:
        print("⚠ Model bulunamadı, önce eğitmeniz gerekiyor.")
        return None

# -----------------------------
# 5️⃣ Tek satır tahmin fonksiyonu
# -----------------------------
def make_prediction(model, user_input: dict):
    user_data = pd.DataFrame([user_input])
    
    categorical_cols = ["state_name","date","no2","avg_o3","avg_hcho","avg_o2o2","activity_ok"]
    for col in categorical_cols:
        if col not in user_data.columns:
            user_data[col] = "none"
    user_data = pd.get_dummies(user_data, columns=categorical_cols)
    
    # Eksik kolonları ekle
    missing_cols = set(model.feature_names_in_) - set(user_data.columns)
    for col in missing_cols:
        user_data[col] = 0
    user_data = user_data[model.feature_names_in_]
    
    prediction = model.predict(user_data)[0]
    probability = max(model.predict_proba(user_data)[0])
    status = "Uygun" if prediction == 1 else "Uygun Değil"
    
    return {"durum": status, "probability": round(probability, 2)}

# -----------------------------
# 6️⃣ Toplu tüm veri tahmini
# -----------------------------

    
def predict_all_states(model, csv_path="training_data.csv"):
    data = pd.read_csv(csv_path)
    
    # Kolon adlarını normalize et
    data.columns = [c.strip() for c in data.columns]
    
    if "activity_ok" in data.columns:
        show_activity_ok = True
    else:
        print("⚠ CSV'de 'activity_ok' kolonu bulunamadı, toplu tahmin yapılacak ama activity_ok gösterilemeyecek.")
    show_activity_ok = False 

    X_test = data.drop("activity_ok", axis=1, errors="ignore")
    
    categorical_cols = ["state_name", "date"]
    for col in categorical_cols:
        if col not in X_test.columns:
            X_test[col] = "none"
    
    X_test = pd.get_dummies(X_test, columns=categorical_cols)
    
    missing_cols = set(model.feature_names_in_) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[model.feature_names_in_]
    
    predictions = model.predict(X_test)
    data["activity_predicted"] = predictions
    
    if show_activity_ok:
        return data[["state_name", "activity_ok", "activity_predicted"]]
    else:
        return data[["state_name", "activity_predicted"]]
        

    



# -----------------------------
# 7️⃣ Ana Test Bloğu
# -----------------------------
if __name__ == "__main__":
    # 1️⃣ Modeli CSV'den eğit
    model = train_model_from_csv()

    # 2️⃣ Modeli yükle
    model = load_model()

    print("🔹 Başlangıç")

    model = train_model_from_csv()
    print("🔹 Model eğitildi")

    model = load_model()
    print("🔹 Model yüklendi:", model)
    # 3️⃣ Tek satır test
    test_input = {
        "state_name": "New Mexico",
        "date": "2025-09-25",
        "no2": 20,
        "avg_o3": 15,
        "avg_hcho": 5,
        "avg_o2o2": 30,
        "age_group": "adult",
        "pregnancy_status": "none",
        "respiratory_disease": "no",
        "cardio_disease": "no"
    }
    result = make_prediction(model, test_input)
    print("🔮 Tek Satir Tahmin Sonucu:", result)

    # 4️⃣ Tüm bölgeler için toplu tahmin
    all_results = predict_all_states(model)
    print("🔮 Tüm Bölgeler Tahmin Sonucu:")
    print(all_results)
'''


# predictor.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mysql.connector
from mysql.connector import Error
import os

# -----------------------------
# 1️⃣ Model Eğitimi CSV'den
# -----------------------------
def train_model_from_csv(csv_path=None, model_path="aqi_model.pkl"):
    """
    CSV'den veri alıp modeli eğitir ve kaydeder.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "training_data.csv")

    data = pd.read_csv(csv_path)
    print("CSV yüklendi, kolonlar:", data.columns.tolist())

    X = data.copy()  # artık activity_ok yok, tüm veri X
    # Kategorik kolonlar için one-hot encode
    categorical_cols = ["state_name", "date"]
    for col in categorical_cols:
        if col not in X.columns:
            X[col] = "none"
    X = pd.get_dummies(X, columns=categorical_cols)

    # Train/test split (y olmadığından dummy y ile eğiteceğiz)
    y_dummy = [1]*len(X)  # gerçek target artık yok, sadece RandomForest çalışsın diye
    X_train, X_test, y_train, y_test = train_test_split(X, y_dummy, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print(f"✅ Model CSV'den başarıyla eğitildi ve kaydedildi: {model_path}")
    return model

# -----------------------------
# 2️⃣ MySQL'den veri çekme
# -----------------------------
def load_data_from_mysql(host, user, password, database, table):
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        query = f"SELECT * FROM {table};"
        data = pd.read_sql(query, con=connection)
        connection.close()
        print(f"✅ MySQL'den veri çekildi: {table}")
        return data
    except Error as e:
        print(f"❌ MySQL bağlantı hatası: {e}")
        return None

# -----------------------------
# 3️⃣ MySQL verisi ile model eğitimi
# -----------------------------
def train_model_from_mysql(host, user, password, database, table, model_path="aqi_model.pkl"):
    data = load_data_from_mysql(host, user, password, database, table)
    if data is None or data.empty:
        print("❌ MySQL verisi boş veya çekilemedi.")
        return None

    X = data.copy()  # activity_ok yok
    categorical_cols = ["state_name", "date", "no2", "avg_o3", "avg_hcho", "avg_o2o2"]
    for col in categorical_cols:
        if col not in X.columns:
            X[col] = "none"
    X = pd.get_dummies(X, columns=categorical_cols)

    y_dummy = [1]*len(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_dummy)

    joblib.dump(model, model_path)
    print(f"✅ Model MySQL verisi ile eğitildi ve kaydedildi: {model_path}")
    return model

# -----------------------------
# 4️⃣ Modeli yükleme
# -----------------------------
def load_model(model_path="aqi_model.pkl"):
    try:
        model = joblib.load(model_path)
        print("📂 Model başarıyla yüklendi.")
        return model
    except FileNotFoundError:
        print("⚠ Model bulunamadı, önce eğitmeniz gerekiyor.")
        return None

# -----------------------------
# 5️⃣ Tek satır tahmin fonksiyonu
# -----------------------------
def make_prediction(model, user_input: dict):
    user_data = pd.DataFrame([user_input])

    categorical_cols = ["state_name","date","no2","avg_o3","avg_hcho","avg_o2o2"]
    for col in categorical_cols:
        if col not in user_data.columns:
            user_data[col] = "none"
    user_data = pd.get_dummies(user_data, columns=categorical_cols)

    # Eksik kolonları ekle
    missing_cols = set(model.feature_names_in_) - set(user_data.columns)
    for col in missing_cols:
        user_data[col] = 0
    user_data = user_data[model.feature_names_in_]

    prediction = model.predict(user_data)[0]
    probability = max(model.predict_proba(user_data)[0])
    status = "Uygun" if prediction == 1 else "Uygun Değil"

    return {"durum": status, "probability": round(probability, 2)}

# -----------------------------
# 6️⃣ Toplu tüm veri tahmini
# -----------------------------
def predict_all_states(model, csv_path="training_data.csv"):
    data = pd.read_csv(csv_path)
    data.columns = [c.strip() for c in data.columns]
    return data  # artık sadece CSV verisini döndürür, activity_ok yok

def make_prediction(model, user_input: dict):
    import pandas as pd
    user_data = pd.DataFrame([user_input])

    categorical_cols = ["state_name","date","no2","avg_o3","avg_hcho","avg_o2o2"]
    for col in categorical_cols:
        if col not in user_data.columns:
            user_data[col] = "none"
    user_data = pd.get_dummies(user_data, columns=categorical_cols)

    # Eksik kolonları ekle
    missing_cols = set(model.feature_names_in_) - set(user_data.columns)
    for col in missing_cols:
        user_data[col] = 0
    user_data = user_data[model.feature_names_in_]

    prediction = model.predict(user_data)[0]
    probability = max(model.predict_proba(user_data)[0])
    status = "Uygun" if prediction == 1 else "Uygun Değil"

    return {"durum": status, "probability": round(probability, 2)}



# -----------------------------
# 7️⃣ Ana Test Bloğu
# -----------------------------
if __name__ == "__main__":
    model = train_model_from_csv()
    model = load_model()

    print("🔹 Başlangıç")

    model = train_model_from_csv()
    print("🔹 Model eğitildi")

    model = load_model()
    print("🔹 Model yüklendi:", model)

    test_input = {
        "state_name": "New Mexico",
        "date": "2025-09-25",
        "no2": 20,
        "avg_o3": 15,
        "avg_hcho": 5,
        "avg_o2o2": 30,
        "age_group": "adult",
        "pregnancy_status": "none",
        "respiratory_disease": "no",
        "cardio_disease": "no"
    }
    result = make_prediction(model, test_input)
    print("🔮 Tek Satir Tahmin Sonucu:", result)

    all_results = predict_all_states(model)
    print("🔮 Tüm Bölgeler Tahmin Sonucu:")
    print(all_results)
