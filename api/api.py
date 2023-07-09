import cv2
import io
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from keras.models import load_model
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Classify fresh/rotten fn
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

    # img data to a np arr and read using cv2
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Cnvrt BGR to RGB & resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))

    # Preprocess the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def evaluate_rotten_vs_fresh(image_path):
    # Load and predict using the model
    model = load_model('api/rottenvsfresh98pval.h5')
    prediction = model.predict(pre_proc_img(image_path))

    return prediction[0][0]


def ident_type(img): #identify type of fruit/veg using pytorch
    # Load the pretrained model
    model = models.mobilenet_v2(weights=None)
    num_classes = 36  # Update with the number of classes in your model
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load('api/modelforclass.pth', map_location=torch.device('cpu')))
    model.eval()

    # Define the data transforms and class labels
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    class_labels = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
                    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi',
                    'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate',
                    'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
                    'watermelon']
    with torch.no_grad():
        img = transform(img)
        img = img.unsqueeze(0)
        output = model(img)

    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_labels[predicted_idx.item()]
    return predicted_label


@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image = request.files['image']
    img = Image.open(image)
    is_fresh = 1 - evaluate_rotten_vs_fresh(img)
    pred_type = ident_type(img)
    return jsonify({'prediction': str(is_fresh), 'freshness':ret_fresh(is_fresh), 'type':pred_type})


if __name__ == '__main__':
    app.run(host='::', port=6000)