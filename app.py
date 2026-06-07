import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load Trained Model
model = tf.keras.models.load_model("herbal_model.h5")

# Load Class Labels
with open("class_labels.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Load Plant Descriptions from JSON
with open("plants.json", "r") as f:
    plant_descriptions = json.load(f)

# Upload Folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def predict_plant(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    plant_name = class_labels[class_index]

    # Get plant details from the plant descriptions
    plant_info = plant_descriptions.get(plant_name, {
        "common_name": "Not available",
        "scientific_name": "Not available",
        "benefits": "Not available",
        "habitat": "Not available",
        "traditional_uses": "Not available",
        "parts_used": "Not available",
        "precautions": "Not available",
        "description": "Not available"
    })

    return plant_name, plant_info




@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Get Prediction
            plant_name, plant_info = predict_plant(filepath)

            return jsonify({
                "plant_name": plant_name,
                "common_name": plant_info["common_name"],
                "scientific_name": plant_info["scientific_name"],
                "benefits": plant_info["benefits"],
                "habitat": plant_info["habitat"],
                "traditional_uses": plant_info["traditional_uses"],
                "parts_used": plant_info["parts_used"],
                "precautions": plant_info["precautions"],
                'description':plant_info['description'],
                "image_path": filepath
            })

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)