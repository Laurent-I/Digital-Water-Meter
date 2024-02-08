import cv2
import os
import pytesseract

# Set the tesseract path to the location of your tesseract.exe file
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"

# Load the image from file
image_path = r"Screenshot 2024-01-15 210808.png"
# print(image_path)
image = cv2.imread(image_path)

# Check if the file exists and the script has read permissions
if os.access(image_path, os.R_OK):
    # Load the image from file
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use Tesseract to extract text
    extracted_text = pytesseract.image_to_string(thresholded_image)

    # Print the extracted text
    print(extracted_text)
    
else:
    print(f"Cannot access file: {image_path}")
