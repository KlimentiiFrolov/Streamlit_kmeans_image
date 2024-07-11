import streamlit as st
import pandas as pd
from io import StringIO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


# A function that changes the color of a picture using the k-means algorithm
def color_image_desert(photo, variant):
    colors = [
        [[239, 118, 122], [69, 105, 144], [73, 190, 170]],
        [[255, 33, 140], [255, 216, 0], [33, 177, 255]],
        [[233, 215, 88], [41, 115, 115], [255, 133, 82]],
        [[254, 33, 139], [254, 215, 0], [33, 176, 254]],
    ]
    random_number = random.randint(0, 3)
    convert_photo = photo.reshape((-1, 3))
    convert_photo = np.float32(convert_photo)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    number_of_clusers = 3
    ret, label, center = cv2.kmeans(
        convert_photo, number_of_clusers, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Recoloring pictures according to the selected condition
    if variant == "original":
        center = np.uint8(center)
    elif variant == "desert":
        center = np.array([[13, 59, 102], [250, 240, 202], [244, 211, 94]])
    elif variant == "random":
        center = np.array(colors[random_number])

    # Creating the Final Image
    result = center[label.flatten()]
    result = result.reshape((photo.shape))
    return result[:, :, ::-1]


# Creating a sidebar and title with text
st.title("Paint your photo")
st.sidebar.title("About")
st.sidebar.info(
    """
    This app can change the color of your photo.
    """
)
st.sidebar.info(
    """
    original - original color.
    
    desert - desert color
    
    random - random color
    """
)

# Creating Button to select the color mode
choose = st.radio(
    "Select color",
    [":Documentary original", ":Documentary desert", ":Documentary random"],
)

# Uploads an image
uploaded_file = st.file_uploader("Choose an image file", type="jpg")

# Output of a recolored image according to the condition
placeholder = st.empty()
with placeholder.container():
    if uploaded_file is not None:
        # Convert the uploaded file to normal image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        if choose == ":Documentary original":
            st.image(color_image_desert(opencv_image, "original"), channels="BGR")
        elif choose == ":Documentary desert":
            st.image(color_image_desert(opencv_image, "desert"), channels="BGR")
        elif choose == ":Documentary random":
            st.image(color_image_desert(opencv_image, "random"), channels="BGR")
