import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pretrained model
model = models.mobilenet_v2(weights=None)
num_classes = 36  # Update with the number of classes in your model
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load('modelforclass.pth', map_location=torch.device('cpu')))
model.eval()

# Define the data transformations
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

# Define the API endpoint
@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image = request.files['image']
    img = Image.open(image)
    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_labels[predicted_idx.item()]

    return jsonify({'classification': predicted_label})


if __name__ == '__main__':
    app.run(host='::', port=5000)