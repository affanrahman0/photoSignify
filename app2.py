from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model = load_model('our_model.h5')


# Preprocess the image
def preprocess_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (224, 224))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, 224, 224, 1))
    return img_reshaped


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)[0][0]
        #label = "signature" if prediction[0][0] >= 0.5 else "face"
        #return jsonify({'prediction': label})
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run()
