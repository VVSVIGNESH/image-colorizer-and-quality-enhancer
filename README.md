# image-colorizer-and-quality-enhancer
# Developed an AI-powered image processing tool that uses a Deep learning colorization model  (Caffe) with 86% accuracy to restore colors in black-and-white images. Implemented a Wiener filter-based deblurring algorithm with Fourier transforms to enhance image  clarity and used PyTesseract and OCR to extract text from image.
# In order to run the demo, you will first need to download the pre-trained data from this location. Place the file in the folder with this readme.

# https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1
Image Enhancer:
This project provides an image enhancement tool that performs the following tasks:

Extracts text from black-and-white images using Tesseract OCR.
Checks if the image is blurry and applies deblurring if necessary.
Colorizes black-and-white images using a deep learning model.

Features:
Text Extraction: Extracts text from the image using Tesseract OCR.
Blurriness Check: Detects if the image is blurry and applies deblurring using a Wiener filter.
Colorization: Converts black-and-white images to color using a pre-trained deep learning model.

Prerequisites:
Python 3.x
Tesseract-OCR (Install from here)
OpenCV
NumPy
PyTesseract
Matplotlib

Installation:
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/image-enhancer.git
cd image-enhancer
Install the required Python packages:
Create a requirements.txt file and add the following:

Copy code
numpy
opencv-python
pytesseract
matplotlib
Then run:

bash
Copy code
pip install -r requirements.txt
Install Tesseract-OCR:

Download and install Tesseract from Tesseract OCR GitHub.
Ensure that the Tesseract executable path is correctly set in your script.
Usage
Prepare Your Image:
Ensure that the image file is named bw.jpg or update the path in the script accordingly.

Run the Script:
Execute the Python script:

bash
Copy code
python image_enhancer.py
Results:

Extracted Text: Displays the text extracted from the image.
Original Image: Shows the original black-and-white image.
Deblurred Image: Shows the deblurred image (if applicable).
Colorized Image: Displays the colorized version of the image.

Troubleshooting:
Tesseract Not Found: Ensure the path to the Tesseract executable is correctly set in the script and that Tesseract is installed properly.
Missing Files: Ensure that the model files (colorization_deploy_v2.prototxt, pts_in_hull.npy, colorization_release_v2.caffemodel) are present in the same directory as the script.

Contributing:
Contributions are welcome! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

License:
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments:
The Tesseract OCR tool for text extraction.
OpenCV for image processing.
The deep learning model for image colorization.
