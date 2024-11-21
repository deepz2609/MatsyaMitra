from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load the pre-trained model (make sure the model path is correct)
model = tf.keras.models.load_model('fish.h5')

# Define image size (adjust this based on your model)
IMG_SIZE = (224, 224)

def prepare_image(image_path):
    """
    Preprocess the image for prediction.
    This should match the image preprocessing used during training.
    """
    img = Image.open(image_path)
    
    # Remove background (assuming a white background)
    img = img.convert("RGBA")
    datas = img.getdata()

    new_data = []
    for item in datas:
        # Change all white (also shades of whites)
        # pixels to transparent
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)
    img = img.convert("RGB")
    
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file temporarily
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Prepare the image and make prediction
    img = prepare_image(filepath)
    prediction = model.predict(img)

    # Assuming your model has classes in a list, e.g., ['class1', 'class2', 'class3']
    class_names = ['Pomfret', 'Mackerel', 'Black Snapper', 'Indian Carp', 'Prawn', 'Pink Perch', 'Indian Carp', 'Black Pomfret']  # Adjust to your model's classes

    # Get the index of the highest prediction
    predicted_index = np.argmax(prediction)

    # Check if the predicted index is within the range of class_names
    if predicted_index < len(class_names):
        predicted_class = class_names[predicted_index]
    else:
        return jsonify({'error': 'Prediction index out of range'})

    # Return the result
    # return jsonify({'prediction': predicted_class})
    
    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)