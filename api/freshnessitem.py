import cv2
import io
import numpy as np
from keras.models import load_model
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Classify fresh/rotten
def ret_fresh(res):
    threshold_fresh = 0.90  # set according to standards
    threshold_medium = 0.50  # set according to standards
    if res > threshold_fresh:
        return "The item is VERY FRESH!"
    elif threshold_fresh > res > threshold_medium:
        return "The item is FRESH"
    else:
        return "The item is NOT FRESH"


def pre_proc_img(image_data):
    # Convert the JpegImageFile object to bytes
    byte_stream = io.BytesIO()
    image_data = image_data.convert('RGB')  # Convert to RGB
    image_data.save(byte_stream, format='JPEG')
    image_bytes = byte_stream.getvalue()

    # Convert the image data to a numpy array
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Read the image using OpenCV
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image
    img = cv2.resize(img, (100, 100))

    # Preprocess the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def evaluate_rotten_vs_fresh(image_path):
    # Load the trained model
    model = load_model('rottenvsfresh98pval.h5')

    # Read and process and predict
    prediction = model.predict(pre_proc_img(image_path))

    return prediction[0][0]

# Define the API endpoint
@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image = request.files['image']
    img = Image.open(image)
    is_fresh = 1 - evaluate_rotten_vs_fresh(img)
    return jsonify({'prediction': str(is_fresh), 'freshness':ret_fresh(is_fresh)})

if __name__ == '__main__':
    app.run(host='::', port=5000)