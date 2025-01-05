import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Model paths
MODEL_PATH = "model.tflite"
CLASS_INDICES_PATH = "class_indices.json"

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Load class indices
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
else:
    class_indices = {}

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    img = Image.open(BytesIO(image)).resize(target_size)
    img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
    return img_array

# Prediction function using TensorFlow Lite
def predict_plant_disease(image):
    preprocessed_img = preprocess_image(image)
    
    # Set up the input and output tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor data
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)

    # Run inference
    interpreter.invoke()

    # Get the output
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices.get(str(predicted_class_index), "Unknown Class")

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Image classification route
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        image_data = file.read()
        prediction = predict_plant_disease(image_data)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
