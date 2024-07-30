import streamlit as st
import pandas as pd
from io import StringIO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# constants
DESERT_COLOR = np.array([[13, 59, 102], [250, 240, 202], [244, 211, 94]])
SALAT = np.array([[239, 118, 122], [69, 105, 144], [73, 190, 170]])
ORANGE = np.array([[233, 215, 88], [41, 115, 115], [255, 133, 82]])
SKY = np.array([[255, 255, 255], [255, 202, 212], [176, 208, 211]])
DIRTY = np.array([[208, 184, 172], [243, 216, 199], [239, 229, 220]])


# A function that changes the color of a picture using the k-means algorithm
def color_image_desert(photo: np.ndarray, variant: str) -> np.ndarray:
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
        center = DESERT_COLOR
    elif variant == "salat":
        center = SALAT
    elif variant == "orange":
        center = ORANGE
    elif variant == "sky":
        center = SKY
    elif variant == "dirty":
        center = DIRTY

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
    
    salat - salat color
    
    orange - orange color
    
    sky - sky color
    
    dirty - dirty color
    """
)

# Creating Button to select the color mode
choose = st.radio(
    "Select color",
    ["original", "desert", "salat", "orange", "sky", "dirty"],
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
        if choose == "original":
            st.image(color_image_desert(opencv_image, "original"), channels="BGR")
        elif choose == "desert":
            st.image(color_image_desert(opencv_image, "desert"), channels="BGR")
        elif choose == "salat":
            st.image(color_image_desert(opencv_image, "salat"), channels="BGR")
        elif choose == "orange":
            st.image(color_image_desert(opencv_image, "orange"), channels="BGR")
        elif choose == "sky":
            st.image(color_image_desert(opencv_image, "sky"), channels="BGR")
        elif choose == "dirty":
            st.image(color_image_desert(opencv_image, "dirty"), channels="BGR")
