import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Install Tesseract
!sudo apt-get install tesseract-ocr

# Set up Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Load the colorization model
PROTOTXT = "colorization_deploy_v2.prototxt"
POINTS = "pts_in_hull.npy"
MODEL = "colorization_release_v2.caffemodel"

# Load the model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Helper functions
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def deblur_image(image):
    h, w = image.shape[:2]
    y, x = np.mgrid[:h, :w]
    f = 0.1
    sigma = 0.01
    PSF = np.exp(-0.5 * ((x - w // 2)**2 + (y - h // 2)**2) / (f**2)) / (2 * np.pi * f**2)
    PSF /= PSF.sum()
    img_float = image.astype(np.float32) / 255.0
    img_fft = np.fft.fft2(img_float, axes=(0, 1))
    PSF_fft = np.fft.fft2(PSF, s=img_float.shape[:2], axes=(0, 1))
    img_deblur_fft = img_fft / (PSF_fft + sigma)
    img_deblur = np.fft.ifft2(img_deblur_fft, axes=(0, 1))
    img_deblur = np.abs(np.fft.ifftshift(img_deblur))
    return np.clip(img_deblur * 255, 0, 255).astype(np.uint8)

def is_black_and_white(image):
    if len(image.shape) == 2:
        return True
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 1], image[:, :, 2])
    return False

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray).strip()
    extracted_text = text if text else "No text detected in the image"

    if is_blurry(image):
        deblurred_image = deblur_image(image)
    else:
        deblurred_image = image

    if not is_black_and_white(image):
        return extracted_text, deblurred_image, None

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return extracted_text, deblurred_image, colorized

# Streamlit App
st.title("Image Colorization and Text Extraction")
st.write("Upload an image to colorize and extract text.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))

    st.image(image, caption="Uploaded Image", use_column_width=True)

    extracted_text, deblurred_image, colorized_image = process_image(image)

    st.write("**Extracted Text:**")
    st.text(extracted_text)

    if deblurred_image is not None:
        st.image(deblurred_image, caption="Deblurred Image", use_column_width=True)

    if colorized_image is not None:
        st.image(colorized_image, caption="Colorized Image", use_column_width=True)
    else:
        st.write("The image is already colored, so no colorization was applied.")
