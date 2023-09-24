import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Corrected function for loading the model
def load_model():
    #model = tf.keras.models.load_model('hackomedfinaaaal.h5')
    model = tf.keras.models.load_model('EfficientNetB2-Skin-87.h5')
    print("Model loaded")
    return model

model = load_model()

def predict(image):
    image = image.resize((224, 224))  # Resize to your model's input size
    # Convert the image to an array and preprocess it
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = preprocess_input(image_array)  # Preprocess using the same function as during training
    # Make a prediction
    predictions = model.predict(image_array)
    # Interpret the prediction (adjust this based on your model's output)
    predicted_class = np.argmax(predictions[0])  # Assuming your model outputs class probabilities
    # Print the prediction
    #class_names = ["hazardous", "inorganic", "organic"]
    class_names = ['Eczema', 'Warts Molluscum and other Viral Infections', 'Melanoma', 'Atopic Dermatitis',
    'Basal Cell Carcinoma (BCC)', 'Melanocytic Nevi (NV)', 'Benign Keratosis-like Lesions (BKL)',
    'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors',
    'Tinea Ringworm Candidiasis and other Fungal Infections']
    predicted_probability = predictions[0][predicted_class]  # Probability of the predicted class
    response = {"skin_disease": class_names[predicted_class], "probability": float(predicted_probability)}
    return response
