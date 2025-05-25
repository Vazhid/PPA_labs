import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sklearn.metrics import f1_score, confusion_matrix

# Загрузка модели и тестовых данных
try:
    model_path = 'fedot_model.pkl'
    fedot_model = joblib.load(model_path)
    
    # Загрузка тестовых данных (предполагаем, что они сохранены в CSV)
    test_data = pd.read_csv('test_data.csv')
    X_test = test_data.drop('target', axis=1).values
    y_test = test_data['target'].values
except Exception as e:
    print(f"Error loading model or data: {e}")
    raise

# Создание приложения FastAPI
app = FastAPI(title="ML Model API")

# Добавляем корневой эндпоинт
@app.get("/")
async def read_root():
    return {"message": "Welcome to the ML Model API"}

# Определение модели данных
class DataInput(BaseModel):
    features: list  # Список с признаками

# Создание endpoint для получения предсказания
@app.post("/predict/")
async def predict(data: DataInput):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = fedot_model.predict(features=features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Пример endpoint для получения метрик F1 и матрицы ошибок
@app.get("/metrics/")
async def get_metrics():
    try:
        fedot_preds = fedot_model.predict(features=X_test)
        fedot_f1_macro = f1_score(y_test, fedot_preds, average='macro')
        fedot_f1_weighted = f1_score(y_test, fedot_preds, average='weighted')
        fedot_cm = confusion_matrix(y_test, fedot_preds)

        return {
            "F1 Macro": float(fedot_f1_macro),
            "F1 Weighted": float(fedot_f1_weighted),
            "Confusion Matrix": fedot_cm.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 