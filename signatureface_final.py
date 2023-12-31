# -*- coding: utf-8 -*-
"""signatureface_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fUpHULMSoDvUZiAoMhjOstS2Q1zuzA6Y
"""

import tensorflow as tf
from PIL import Image
import numpy as np
import os
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Set the path to the labeled training dataset
face_dataset_path = '/content/drive/MyDrive/f2'
signature_dataset_path = '/content/drive/MyDrive/s2_final'

# Define and compile the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Preprocess and load the training dataset
train_images = []
train_labels = []

# Load face images
for image_file in os.listdir(face_dataset_path):
    image_path = os.path.join(face_dataset_path, image_file)
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    train_images.append(img_array)
    train_labels.append(0)  # Assign label 0 for face images

# Load signature images
for image_file in os.listdir(signature_dataset_path):
    image_path = os.path.join(signature_dataset_path, image_file)
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    train_images.append(img_array)
    train_labels.append(1)  # Assign label 1 for signature images

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Perform random undersampling on the majority class
majority_indices = np.where(train_labels == 0)[0]
minority_indices = np.where(train_labels == 1)[0]

undersampled_majority_indices = resample(
    majority_indices, replace=False, n_samples=len(minority_indices), random_state=42
)

undersampled_indices = np.concatenate([undersampled_majority_indices, minority_indices])
train_images = train_images[undersampled_indices]
train_labels = train_labels[undersampled_indices]

print(undersampled_majority_indices.shape)
print(minority_indices.shape)

# Train the CNN model
model.fit(train_images, train_labels, epochs=10)

from sklearn.metrics import classification_report
def evaluate():
    # Perform predictions on the test dataset
    predictions = model.predict(test_images)
    predictions = (predictions >= 0.5).astype(int)

    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels)
    print("Accuracy:", accuracy)

    # Generate classification report
    report = classification_report(test_labels, predictions)
    print("Classification Report:\n", report)

    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix:")
    print(cm)

evaluate()

# Save the trained model
model.save_weights('signature_classifier_weights.h5')

# from keras.models import Sequential
from keras.layers import Dense
model.save('model.h5')

import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report
# keras.models import load_model

# Load the model
model = load_model('our_model.h5')

# Load the pre-trained model
#model = load_model('my_model.h5')

# Load the image and convert it to grayscale
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (224, 224))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, 224, 224, 1))
    return img_reshaped

def predict(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    #return prediction[0][0]
    # Convert the prediction to a human-readable label
    label = "signature" if prediction[0][0] >= 0.5 else "face"
    #print(label)
    return label

image_path = '/content/drive/MyDrive/s2/NFI-00102027.PNG'  # Specify the path to the image you want to predict
prediction = predict(image_path)
print("Prediction:", prediction)