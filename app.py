from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
from PIL import Image
import os

app = Flask(__name__)

# ------------------------
# Folder settings
# ------------------------
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------
# Load AI Model
# ------------------------
device = torch.device("cpu")

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: No Defect (0) / Defect (1)
model.load_state_dict(torch.load("model/defecscan_resnet18_binary.pth", map_location=device))
model.eval()

# ------------------------
# Image Transformation
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------
# ROUTES
# ------------------------

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# About Us Page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact Us Page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Upload and Predict
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded!", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file!", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and Transform the Image
    image = Image.open(filepath).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Model Prediction
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
        result = "✅ No Defect Detected" if prediction.item() == 0 else "⚠️ Defect Detected"

    return render_template('index.html', result=result, image=filename)

# ------------------------

# Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
