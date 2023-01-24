# imports
import streamlit as st
import pandas as pd
import cv2
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
from rembg import remove
from PIL import Image

# -------------------------------------------------------------------------------------
# Page Configuration:

st.set_page_config(page_title="Is My Banana Ripe?",page_icon=":banana:",layout="wide")

# -------------------------------------------------------------------------------------
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
    loaded_image = st.file_uploader("Upload an image of a banana:", type=['jpg', 'jpeg', 'png'])
    if loaded_image is not None:
        image = Image.open(loaded_image)
        st.image(image, caption='Uploaded banana')
    else:
        st.markdown(f'<h1 style="color:red;font-size:24px;">{"Uploading an image is required"}</h1>', unsafe_allow_html=True)
        
# -------------------------------------------------------------------------------------
# Banana Ripeness Color Scheme:
UNRIPE = np.array([[143, 169, 65]])
SEMI_RIPE = np.array([[213, 209, 122]])
RIPE = np.array([[244, 219, 100]])
HALF_ROTTEN = np.array([[234, 171, 64]])
ROTTEN = np.array([[46, 37, 34]])

# Grayscale Color Scheme
G_UNRIPE = np.array([146])
G_SEMI_RIPE = np.array([200])
G_RIPE = np.array([213])
G_HALF_ROTTEN = np.array([178])
G_ROTTEN = np.array([39])

# -------------------------------------------------------------------------------------
# steps 1+2: reading the banana image and removing the background using rembg

output = remove(image) # remove background
#output.save(output_path) # save image

# -------------------------------------------------------------------------------------
# step 3: pre-process the image before segmentation

bg_removed = np.array(output) # convert image to numpy array

new  = bg_removed[:,:,:3] # removing the alpha channel (converting from RGBA to RGB)

bg_removed_gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY) # converting to grayscale

# -------------------------------------------------------------------------------------
# step 4: K-means segmentation

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = bg_removed_gray.reshape((-1, 1))
# convert to float
pixel_values = np.float32(pixel_values)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 6
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(bg_removed_gray.shape)

# -------------------------------------------------------------------------------------
# step 5: getting the centers of the clusters (the weighted average of the pixels in each cluster)

# count how many pixels are in each cluster
pixel_counts = np.bincount(labels)

# convert ceneters to a list of integers
centers = centers.tolist()

# convert each element in the list to an integer (from a list of lists)
for i in range(len(centers)):
    centers[i] = int(centers[i][0])

# create a dataframe of the pixel counts, the centers and their weighted average
df = pd.DataFrame({'centers': centers, 'pixel_counts':pixel_counts, 'real_product':pixel_counts*centers})

# remove the line with the [0] center (the background)
df = df.drop(df[df.centers==0].index)

# -------------------------------------------------------------------------------------
# step 6: classifying each center to its closest grayscale value and calculating other values

grayscale_values = [146, 200, 213, 178, 39] # the grayscale values of the colors

def closest(lst, K): # function to find the closest grayscale value to each center
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

df['closest_grayscale'] = df['centers'].apply(lambda x: closest(grayscale_values, x)) # apply the function to the centers

# calculating the desired product
df['desired_product'] = df['pixel_counts']*df['closest_grayscale']

# adding a column with the percentage of the total pixels in each cluster
df['percentage'] = df['pixel_counts']/df['pixel_counts'].sum()*100

# changing the order of the columns
df = df[['centers', 'closest_grayscale', 'pixel_counts','percentage', 'real_product', 'desired_product']]

# creating a new datafram (df2) with everything but thr centers column
df2 = df.drop(['centers'], axis=1)

# add a column with the ripening level
df2['ripening_level'] = df2['closest_grayscale'].apply(lambda x: 'unripe' if x == 146 else 'semi-ripe' if x == 200 else 'ripe' if x == 213 else 'half-rotten' if x == 178 else 'rotten')

# group by ripening level and sum the percentages
df2 = df2.groupby(['ripening_level']).sum()

# -------------------------------------------------------------------------------------
# step 7: calculating the ripeness status
# the nearest of the desired products to the average of the real products is the ripeness status

# the average of the real products
real_product_avg = df2['real_product'].sum()/len(df2)

# finding the nearest of the desired products to the average of the real products
nearest = closest(df2['desired_product'].tolist(), real_product_avg)

# finding the ripeness status
ripeness_status = df2[df2['desired_product']==nearest].index[0]

# -------------------------------------------------------------------------------------
# Output:

with output:
    st.header(f"Calculated Status: {ripeness_status}")
    st.image(image, caption='Uploaded banana')
    
    # plot ripeness percentage
    st.bar_chart(df2['percentage'])
    plt.title('Ripeness Percentage of the Banana')
    plt.gca().set_ylabel('Ripeness Level')
    plt.gca().set_xlabel('Percentage')
