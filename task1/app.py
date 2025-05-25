from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

app = Flask(__name__)

# Загрузка модели
model = joblib.load('tpot_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics', methods=['POST'])
def get_metrics():
    try:
        data = request.get_json()
        X_test = np.array(data['X_test'])
        y_test = np.array(data['y_test'])
        preds = model.predict(X_test)
        return jsonify({
            'f1_macro': float(f1_score(y_test, preds, average='macro')),
            'f1_weighted': float(f1_score(y_test, preds, average='weighted')),
            'confusion_matrix': confusion_matrix(y_test, preds).tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Для разработки (не для production!)
    app.run(host='0.0.0.0', port=5000)
    
    # Для production используйте:
    # serve(app, host='0.0.0.0', port=5000)