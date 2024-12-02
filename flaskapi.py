import os
import pandas as pd
import numpy as np
import pickle
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# === Paths to datasets and models ===
DATASET_PATH = "./archive5/kag2"
RAINFALL_MODEL_PATH = 'rf_model.pkl'
CROP_MODEL_PATH = 'crop_yield_model.pkl'
YIELD_MODEL_PATH = 'yield.pkl'
RAINFALL_DATA_PATH = 'crop.csv'
CROP_DATA_PATH = 'limited_top_ten_crops.csv'

# === Load Models ===
# Image-based crop classification model
model = load_model("./my_model_66.h5")

# Rainfall prediction model
with open(RAINFALL_MODEL_PATH, 'rb') as file:
    rainfall_model = joblib.load(file)

# Crop recommendation model
crop_model = joblib.load(CROP_MODEL_PATH)

# Yield prediction model
with open(YIELD_MODEL_PATH, 'rb') as file:
    yield_bundle = joblib.load(file)
yield_model = yield_bundle['model']
label_encoder_state = yield_bundle['label_encoder_state']
label_encoder_crop = yield_bundle['label_encoder_crop']

# === Data Preprocessing ===
# Generate class labels for crop classification
def get_class_labels():
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(224, 224),
        batch_size=64,
        class_mode="categorical",
        shuffle=False,
    )
    class_labels = generator.class_indices
    return {v: k for k, v in class_labels.items()}

class_labels = get_class_labels()

# Load rainfall dataset and encoders
rainfall_data = pd.read_csv(RAINFALL_DATA_PATH)
rainfall_label_encoders = {}
for column in ['Crop', 'Season', 'State']:
    le = LabelEncoder()
    rainfall_data[column] = le.fit_transform(rainfall_data[column])
    rainfall_label_encoders[column] = le

# Load crop dataset and encoders
crop_data = pd.read_csv(CROP_DATA_PATH)
crop_data['Season'] = crop_data['Season'].str.strip()
le_state = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()
crop_data['State'] = le_state.fit_transform(crop_data['State'])
crop_data['Season'] = le_season.fit_transform(crop_data['Season'])
crop_data['Crop'] = le_crop.fit_transform(crop_data['Crop'])

# === Image-based Crop Classification Helper Functions ===
def find_file_in_dataset(filename):
    possible_extensions = [".jpeg", ".jpg", ".png"]
    for subfolder in os.listdir(DATASET_PATH):
        subfolder_path = os.path.join(DATASET_PATH, subfolder)
        if os.path.isdir(subfolder_path):
            for ext in possible_extensions:
                full_path = os.path.join(subfolder_path, filename + ext)
                if os.path.exists(full_path):
                    return full_path
    return None

def predict_crop(filepath):
    img = cv2.imread(filepath)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.array(img_resized).reshape((1, 224, 224, 3)) / 255.0
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    return predicted_index, predictions[0][predicted_index]

def generate_prediction_plot(filepath, predicted_label, ground_truth_label):
    img = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"Ground Truth: {ground_truth_label}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("static/prediction_result.png")
    plt.close()

# === Rainfall Prediction Logic ===
def predict_rainfall(year, state):
    try:
        state_encoded = rainfall_label_encoders['State'].transform([state])[0]
        crop_encoded = rainfall_data['Crop'].mode()[0]
        season_encoded = rainfall_data['Season'].mode()[0]
        input_data = pd.DataFrame({
            'Crop': [crop_encoded],
            'Crop_Year': [year],
            'Season': [season_encoded],
            'State': [state_encoded],
            'Area': [rainfall_data['Area'].mean()],
            'Production': [rainfall_data['Production'].mean()],
            'Fertilizer': [rainfall_data['Fertilizer'].mean()],
            'Pesticide': [rainfall_data['Pesticide'].mean()],
            'Yield': [rainfall_data['Yield'].mean()],
        })
        return rainfall_model.predict(input_data)[0]
    except Exception as e:
        raise ValueError(f"Error in predicting rainfall: {e}")

# === Routes ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    filepath = find_file_in_dataset(filename)
    if not filepath:
        return jsonify({"error": f"File {filename} not found in dataset"}), 404
    predicted_index, confidence = predict_crop(filepath)
    predicted_label = class_labels[predicted_index]
    ground_truth_label = os.path.basename(os.path.dirname(filepath))
    generate_prediction_plot(filepath, predicted_label, ground_truth_label)
    return jsonify({
        "ground_truth": ground_truth_label,
        "predicted_label": predicted_label,
        "confidence": float(confidence),
    })

@app.route('/predict_rainfall', methods=['POST'])
def predict_rainfall_route():
    data = request.get_json()
    year = data['year']
    state = data['state']
    prediction = predict_rainfall(year, state)
    return jsonify({'predicted_rainfall': prediction, 'year': year, 'state': state})

@app.route('/predict_top_5_crops', methods=['POST'])
def predict_top_5_crops():
    data = request.json
    state = data['state']
    year = data['year']
    season = data['season']
    area = data['area']
    fertilizer = data['fertilizer']
    pesticide = data['pesticide']
    state_encoded = le_state.transform([state])[0]
    season_encoded = le_season.transform([season])[0]
    rainfall_info = crop_data[(crop_data['State'] == state_encoded) & (crop_data['Crop_Year'] == year)]
    annual_rainfall = rainfall_info['Annual_Rainfall'].mean() if not rainfall_info.empty else 0
    input_data = pd.DataFrame({
        'State': [state_encoded] * len(le_crop.classes_),
        'Crop': range(len(le_crop.classes_)),
        'Crop_Year': [year] * len(le_crop.classes_),
        'Season': [season_encoded] * len(le_crop.classes_),
        'Area': [area] * len(le_crop.classes_),
        'Fertilizer': [fertilizer] * len(le_crop.classes_),
        'Pesticide': [pesticide] * len(le_crop.classes_),
        'Annual_Rainfall': [annual_rainfall] * len(le_crop.classes_),
    })
    probabilities = crop_model.predict_proba(input_data)[:, 1]
    crop_probabilities = pd.DataFrame({
        'Crop': le_crop.inverse_transform(range(len(le_crop.classes_))),
        'High_Yield_Probability': probabilities,
    })
    top_5_crops = crop_probabilities.sort_values(by='High_Yield_Probability', ascending=False).head(5)
    return jsonify(top_5_crops.to_dict(orient='records'))

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    data = request.get_json()
    year = data.get('year')
    state = data.get('state')
    crop = data.get('crop')
    fertilizer = data.get('fertilizer')
    pesticide = data.get('pesticide')
    rainfall = data.get('rainfall')
    state_encoded = label_encoder_state.transform([state])[0]
    crop_encoded = label_encoder_crop.transform([crop])[0]
    input_data = np.array([[year, state_encoded, rainfall, crop_encoded, fertilizer, pesticide]])
    predicted_yield = yield_model.predict(input_data)[0]
    return jsonify({'predicted_yield': round(predicted_yield, 2)})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
