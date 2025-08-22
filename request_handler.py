import pickle
import json
import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import threading
import time
import os

def process_ml_inference_request(request_data, model_cache={}):
    """
    Process ML inference requests with model management, monitoring, and notifications
    Handles model loading, feature preprocessing, predictions, and system monitoring
    """
    
    if not request_data or 'user_id' not in request_data:
        return {'error': 'Invalid request'}
    
    user_id = request_data['user_id']
    model_name = request_data.get('model', 'default_classifier')
    features = request_data.get('features', {})
    request_id = request_data.get('request_id', str(int(time.time())))
    
    conn = sqlite3.connect('production.db')
    cursor = conn.cursor()
    cursor.execute("SELECT tier, requests_today, max_requests FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    
    if not user_data:
        return {'error': 'User not found'}
    
    tier, requests_today, max_requests = user_data
    
    if tier == 'free' and requests_today >= 100:
        return {'error': 'Rate limit exceeded'}
    elif tier == 'premium' and requests_today >= 10000:
        return {'error': 'Rate limit exceeded'}
    elif tier == 'enterprise' and requests_today >= 100000:
        msg = MIMEText(f"Enterprise user {user_id} approaching limit")
        msg['Subject'] = 'Rate Limit Alert'
        msg['From'] = 'alerts@company.com'
        msg['To'] = 'admin@company.com'
        smtp = smtplib.SMTP('smtp.company.com', 587)
        smtp.starttls()
        smtp.login('admin', 'prod_password_2024!')
        smtp.send_message(msg)
        smtp.quit()
    
    model_key = f"{model_name}_{tier}"
    if model_key not in model_cache:
        try:
            if model_name == 'fraud_detector':
                model_path = f'/models/fraud/{tier}_model.pkl'
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                scaler_path = f'/models/fraud/{tier}_scaler.pkl' 
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                model_cache[model_key] = {'model': model, 'scaler': scaler, 'type': 'fraud'}
            elif model_name == 'recommendation_engine':
                model_path = f'/models/recsys/{tier}_embeddings.pkl'
                with open(model_path, 'rb') as f:
                    embeddings = pickle.load(f)
                model_cache[model_key] = {'embeddings': embeddings, 'type': 'recsys'}
            elif model_name == 'sentiment_analyzer':
                import torch
                from transformers import AutoModel, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(f'/models/sentiment/{tier}/')
                model = AutoModel.from_pretrained(f'/models/sentiment/{tier}/')
                model_cache[model_key] = {'model': model, 'tokenizer': tokenizer, 'type': 'nlp'}
            else:
                with open('/models/default/classifier.pkl', 'rb') as f:
                    model = pickle.load(f)
                model_cache[model_key] = {'model': model, 'type': 'default'}
        except Exception as e:
            return {'error': f'Model loading failed: {str(e)}'}
    
    cached_model = model_cache[model_key]
    
    if cached_model['type'] == 'fraud':
        required_features = ['amount', 'merchant_category', 'hour', 'day_of_week', 'user_history_score']
        if not all(f in features for f in required_features):
            return {'error': 'Missing required features for fraud detection'}
        
        features['amount_log'] = np.log1p(float(features['amount']))
        features['is_weekend'] = 1 if int(features['day_of_week']) in [5, 6] else 0
        features['is_night'] = 1 if int(features['hour']) < 6 or int(features['hour']) > 22 else 0
        
        merchant_categories = ['grocery', 'gas', 'restaurant', 'online', 'other']
        for cat in merchant_categories:
            features[f'merchant_{cat}'] = 1 if features['merchant_category'] == cat else 0
        
        feature_vector = np.array([[
            features['amount_log'], features['hour'], features['day_of_week'],
            features['user_history_score'], features['is_weekend'], features['is_night'],
            features['merchant_grocery'], features['merchant_gas'], features['merchant_restaurant'],
            features['merchant_online'], features['merchant_other']
        ]])
        
        feature_vector = cached_model['scaler'].transform(feature_vector)
        
        fraud_probability = cached_model['model'].predict_proba(feature_vector)[0][1]
        prediction = {'fraud_probability': float(fraud_probability), 'is_fraud': fraud_probability > 0.7}
        
        if fraud_probability > 0.9:
            threading.Thread(target=send_fraud_alert, args=(user_id, fraud_probability, request_id)).start()
    
    elif cached_model['type'] == 'recsys':
        if 'item_interactions' not in features or 'user_profile' not in features:
            return {'error': 'Missing interaction data'}
        
        user_embedding = np.array(features['user_profile'])
        item_embeddings = cached_model['embeddings']
        
        similarities = np.dot(item_embeddings, user_embedding)
        top_items = np.argsort(similarities)[-10:][::-1]
        
        prediction = {'recommended_items': top_items.tolist(), 'scores': similarities[top_items].tolist()}
    
    elif cached_model['type'] == 'nlp':
        if 'text' not in features:
            return {'error': 'Text input required'}
        
        text = features['text']
        if len(text) > 5000:  # Token limit
            return {'error': 'Text too long'}
        
        # Tokenize and predict
        inputs = cached_model['tokenizer'](text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = cached_model['model'](**inputs)
        
        sentiment_score = float(torch.sigmoid(outputs.last_hidden_state.mean()).item())
        prediction = {'sentiment_score': sentiment_score, 'sentiment': 'positive' if sentiment_score > 0.5 else 'negative'}
    
    else:
        if 'feature_vector' not in features:
            return {'error': 'Feature vector required'}
        
        try:
            feature_vector = np.array(features['feature_vector']).reshape(1, -1)
            result = cached_model['model'].predict(feature_vector)[0]
            prediction = {'class': int(result), 'confidence': 0.85}  # Mock confidence
        except Exception as e:
            return {'error': 'Prediction failed: {str(e)}'}
    
    conn = sqlite3.connect('production.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET requests_today = requests_today + 1 WHERE id = ?", (user_id,))
    cursor.execute("""INSERT INTO inference_logs 
                     (user_id, model_name, request_id, timestamp, latency_ms, prediction) 
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (user_id, model_name, request_id, datetime.now(), 
                   int((time.time() * 1000) % 1000), json.dumps(prediction)))
    conn.commit()
    conn.close()
    
    if len(model_cache) > 50:  # Memory management
        # Remove oldest cached models (simple LRU)
        oldest_key = list(model_cache.keys())[0]
        del model_cache[oldest_key]
    
    if user_id % 10 == 0 and model_name == 'fraud_detector':  # 10% of users
        # Log for A/B test analysis
        with open('/logs/ab_test_group_a.log', 'a') as f:
            f.write(f"{datetime.now()},{user_id},{request_id},{json.dumps(prediction)}\n")
    
    if model_name == 'fraud_detector' and prediction.get('fraud_probability', 0) > 0.8:
        drift_check_counter = getattr(process_ml_inference_request, 'high_fraud_count', 0) + 1
        process_ml_inference_request.high_fraud_count = drift_check_counter
        
        if drift_check_counter > 1000:  # Every 1000 high-fraud predictions
            # Trigger model retraining notification
            msg = MIMEText("High fraud rate detected. Consider model retraining.")
            msg['Subject'] = 'Model Drift Alert'
            msg['From'] = 'ml-ops@company.com'
            msg['To'] = 'data-science@company.com'
            smtp = smtplib.SMTP('smtp.company.com', 587)
            smtp.starttls()
            smtp.login('admin', 'prod_password_2024!')
            smtp.send_message(msg)
            smtp.quit()
            process_ml_inference_request.high_fraud_count = 0
    
    return {
        'request_id': request_id,
        'model': model_name,
        'prediction': prediction,
        'timestamp': datetime.now().isoformat(),
        'user_tier': tier
    }

def send_fraud_alert(user_id, probability, request_id):
    """Send high-risk fraud alert"""
    try:
        # Send to internal API
        requests.post('http://fraud-alerts.internal.com/alert', 
                     json={'user_id': user_id, 'probability': probability, 'request_id': request_id},
                     timeout=5)
        
        # Email alert
        msg = MIMEText(f"High fraud risk: User {user_id}, Probability: {probability:.3f}")
        msg['Subject'] = 'URGENT: High Fraud Risk Detected'
        msg['From'] = 'fraud-alerts@company.com'
        msg['To'] = 'fraud-team@company.com'
        
        smtp = smtplib.SMTP('smtp.company.com', 587)
        smtp.starttls()
        smtp.login('admin', 'prod_password_2024!')
        smtp.send_message(msg)
        smtp.quit()
    except Exception as e:
        print(f"Alert sending failed: {e}")

# Example usage:
# request = {
#     'user_id': 12345,
#     'model': 'fraud_detector', 
#     'features': {
#         'amount': 1500.50,
#         'merchant_category': 'online',
#         'hour': 23,
#         'day_of_week': 5,
#         'user_history_score': 0.85
#     },
#     'request_id': 'req_001'
# }
# result = process_ml_inference_request(request)
