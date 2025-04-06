import os
import json
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from preprocess import preprocess_image

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and labels
model = tf.keras.models.load_model('model/food_classifier.h5')
with open('model/food_labels.json', 'r') as f:
    label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_tensor = preprocess_image(filepath)
            pred = model.predict(img_tensor)
            class_idx = pred.argmax()
            prediction = label_map[class_idx]
            confidence = round(float(pred[0][class_idx]) * 100, 2)

    return render_template('index.html', prediction=prediction, confidence=confidence, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
