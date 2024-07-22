# main.py

import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

# Define the model creation function
def create_model():
    # Define the Xception model
    xception_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    xception_out = xception_base.output

    # Define the DenseNet model
    densenet_base = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    densenet_out = densenet_base.output

    # Add custom layers on top
    combined = tf.keras.layers.Concatenate()([xception_out, densenet_out])
    x = tf.keras.layers.GlobalAveragePooling2D()(combined)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='output')(x)  # Ensure the output layer name is unique

    # Create the model
    model = tf.keras.Model(inputs=[xception_base.input, densenet_base.input], outputs=outputs)

    return model

# Define the load_model function
@st.cache_resource
def load_model(path):
    model = create_model()  # Recreate the model structure
    model.load_weights(path)  # Load the model weights
    return model

# Load the Model
model = load_model('/model.h5')

# Title and Description
st.title('Plant Disease Detection')
st.write("Just Upload your Plant's Leaf Image and get predictions if the plant has any disease.")


# Setting the files that can be uploaded
uploaded_file = st.file_uploader("Choose an Image file", type=["png", "jpg", "jpeg"])

# If there is an uploaded file, start making a prediction
if uploaded_file is not None:
    # Display progress and text
    progress = st.text("Crunching Image")
    my_bar = st.progress(0)
    
    # Reading the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    my_bar.progress(30)
    
    # Cleaning the image
    image = clean_image(image)
    my_bar.progress(60)
    
    # Making the predictions
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(90)
    
    # Making the results
    result = make_results(predictions, predictions_arr)
    
    # Show the results
    st.write(f"The plant {result['status']} with {result['prediction']} prediction.")
    
    # Final progress
    my_bar.progress(100)
    st.success("Prediction complete!")
else:
    st.warning("Please upload an image file to get predictions.")
