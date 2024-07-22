import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

# Define a function to load and cache the model
@st.cache(allow_output_mutation=True)
def load_model(path):
    # Define model architecture
    inputs = tf.keras.Input(shape=(512, 512, 3))

    # Xception Model
    xception_base = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
    xception_out = tf.keras.layers.GlobalAveragePooling2D()(xception_base.output)
    xception_out = tf.keras.layers.Dense(4, activation='softmax')(xception_out)
    xception_model = tf.keras.Model(inputs, xception_out)

    # DenseNet Model
    densenet_base = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
    densenet_out = tf.keras.layers.GlobalAveragePooling2D()(densenet_base.output)
    densenet_out = tf.keras.layers.Dense(4, activation='softmax')(densenet_out)
    densenet_model = tf.keras.Model(inputs, densenet_out)

    # Ensembling the Models
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)

    # Average the outputs of the two models
    outputs = tf.keras.layers.Average()([xception_output, densenet_output])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Load the model weights
    model.load_weights(path)

    return model

# Removing Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Loading the Model
model = load_model('model.h5')

# Title and Description
st.title('Plant Disease Detection')
st.write("Just Upload your Plant's Leaf Image and get predictions if the plant is healthy or not")

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
