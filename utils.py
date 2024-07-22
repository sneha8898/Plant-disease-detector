# Imporiting Necessary Libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Clean the image
def clean_image(image):
    image = np.array(image)
    # Resizing the image
    image = np.array(Image.fromarray(image).resize((512, 512), Image.LANCZOS))  # Updated to LANCZOS
    # Adding batch dimensions to the image, taking only the first 3 channels
    image = image[np.newaxis, :, :, :3]
    return image

# Get the prediction from the model
def get_prediction(model, image):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test = datagen.flow(image)
    predictions = model.predict(test)
    predictions_arr = np.argmax(predictions, axis=1)  # Changed to axis=1
    return predictions, predictions_arr

# Make results from the predictions
def make_results(predictions, predictions_arr):
    result = {}
    class_names = ['Healthy', 'Multiple Diseases', 'Rust', 'Scab']  # Ensure these are correct
    prediction_percentage = [f"{int(pred*100)}%" for pred in predictions[0]]
    result = {
        "status": f"has {class_names[int(predictions_arr)]}",
        "prediction": prediction_percentage[int(predictions_arr)]
    }
    return result
