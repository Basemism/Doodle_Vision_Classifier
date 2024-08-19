from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
import io
import torch
import torchvision.transforms as transforms


app = Flask(__name__)
CORS(app)

model = torch.load('model_23c_3000n_10e.pt')
model.eval()

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor(),
    ])
    
    return transform(image).unsqueeze(0)  

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    image_data = request.json['image']
    image_bytes = base64.b64decode(image_data.split(',')[1])
    tensor = transform_image(image_bytes)   
    with torch.no_grad():
        outputs = model(tensor)
        probabilities, indices = torch.topk(torch.nn.functional.softmax(outputs, dim=1), 5)
        guesses = [
            {'index': index.item(), 'probability': prob.item()} 
            for index, prob in zip(indices[0], probabilities[0])
        ]
    
    return jsonify({'guesses': guesses})

if __name__ == '__main__':
    app.run(port=12345, debug=True)
