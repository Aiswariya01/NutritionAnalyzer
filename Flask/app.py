
from flask import Flask, render_template, request
from keras.models import load_model
from keras.utils import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
model = load_model('nutrition.h5')
classes = ['APPLES', 'BANANA', 'ORANGE', 'PINEAPPLE', 'WATERMELON']

# Sample nutrition values
nutrition_data = {
    'APPLES': [
        {"label": "Calories", "value": "95"},
        {"label": "Carbohydrates", "value": "25g"},
        {"label": "Fiber", "value": "4g"}
    ],
    'BANANA': [
        {"label": "Calories", "value": "105"},
        {"label": "Carbohydrates", "value": "27g"},
        {"label": "Potassium", "value": "422mg"}
    ],
    'ORANGE': [
        {"label": "Calories", "value": "62"},
        {"label": "Vitamin C", "value": "70mg"},
        {"label": "Sugar", "value": "12g"}
    ],
    'PINEAPPLE': [
        {"label": "Calories", "value": "82"},
        {"label": "Carbohydrates", "value": "22g"},
        {"label": "Vitamin C", "value": "79mg"}
    ],
    'WATERMELON': [
        {"label": "Calories", "value": "85"},
        {"label": "Water", "value": "92%"},
        {"label": "Sugar", "value": "17g"}
    ]
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    img = load_img(file_path, target_size=(64, 64))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]

    nutrition = nutrition_data.get(predicted_class, [])
    return render_template('imageprediction.html', prediction=predicted_class, img_path=file_path, nutrition=nutrition)

if __name__ == '__main__':
    app.run(debug=True)
