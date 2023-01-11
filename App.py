# imports
import streamlit as st
import cv2
import numpy as np
import skimage.io as io

# ----------------------------

# Functions:
def segment_image_kmeans(img, k=5, attempts=10): 

    # Convert MxNx3 image into Kx3 where K=MxN
    pixel_values  = img.reshape((-1,3))  #-1 reshape means, in this case MxN

    #We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()
    
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)
    
    return segmented_image
# ----------------------------

# Interface:

header = st.container()
input = st.container()
output = st.container()

with header:
    st.title("To Eat, Or Not To Eat?")
    st.subheader("Low-Computing-Power Banana Ripening Calculator")
    st.write("This project was built as a part of \"Intro To Image Processing\"\ncourse in the Faculty of Agriculture.\nIt's quite simple:\n")
    st.write("*  Upload an image of a [banana](https://en.wikipedia.org/wiki/Banana) to the \"Input Image\" section.")
    st.write("*  The ripeness status of the banana will be presented in the \"Calculated Status\" section.")

with input:
    st.header("Input Image:")

    sel_col = st.container()
    disp_col = st.container()


with output:
    st.header("Calculated Status:")
