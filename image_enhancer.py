import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ensure this path is correct

# Paths to load the model
PROTOTXT = "colorization_deploy_v2.prototxt"
POINTS = "pts_in_hull.npy"
MODEL = "colorization_release_v2.caffemodel"

# Load the Modelṇṇṇ
print("Loading model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Function to determine if the image is blurry
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# Function to deblur the image
def deblur_image(image):
    # Using the Wiener filter method for deblurring
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

# Function to check if an image is black and white
def is_black_and_white(image):
    if len(image.shape) == 2:
        return True  # The image is already in grayscale
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Check if all channels are the same
        return np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 1], image[:, :, 2])
    return False

# Load the input image
image_path = "bw.jpg"  # Path to your black and white image
image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    raise ValueError(f"Could not open or find the image: {image_path}")

# Extract text from the original image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(gray)
extracted_text = text if text else "No text in the given image"

print(extracted_text)

# Check if the image is blurry and deblur if necessary
if is_blurry(image):
    print("Image is blurry, applying deblurring.")
    deblurred_image = deblur_image(image)
else:
    print("Image is not blurry, no deblurring applied.")
    deblurred_image = image

# Check if the image is black and white
if not is_black_and_white(image):
    print("The image is not black and white. Exiting.")
    exit()

# Colorizing the image
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")

# Save the results
cv2.imwrite("colorized_image.jpg", colorized)

# Display the images using matplotlib
plt.figure(figsize=(18, 6))

# Display Extracted Text
plt.subplot(1, 4, 1)
plt.title("Extracted Text: ")
plt.text(0.5, 0.5, extracted_text, ha='center', va='center', wrap=True)
plt.axis('off')

# Display Original Image
plt.subplot(1, 4, 2)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Display Deblurred Image
plt.subplot(1, 4, 3)
if is_blurry(image):
    plt.subplot(1, 4, 3)
    plt.title("Deblurred")
    plt.imshow(cv2.cvtColor(deblurred_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
else:
    plt.title("Deblurred: ")
    deblurred_text="Input image is not blurred\n according to the Laplacian Variance method"
    plt.text(0.5, 0.5, deblurred_text, ha='center', va='center', wrap=True)
    plt.axis('off')


# Display Colorized Image
plt.subplot(1, 4, 4)
if is_black_and_white(image):
    plt.title("Colorized", loc="center")
    plt.imshow(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))
    plt.axis('off')
else:
    plt.title("Colorized: ")
    colorized_text="Colorization is not required\n Since the input image\n is already colored"
    plt.text(0.5, 0.5, colorized_text, ha='center', va='center', wrap=True)
    plt.axis('off')
plt.show()
