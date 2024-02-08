import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

img = cv2.imread('Screenshot 2024-02-08 185448.png')

# Set the tesseract path to the location of your tesseract.exe file
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

image_path = r"C:\Users\Admin\Pictures\Screenshots\Screenshot 2024-01-24 160522.png"
image = cv2.imread(image_path)

gray = get_grayscale(image)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

# Perform OCR on the processed images
gray_text = pytesseract.image_to_string(gray)
thresh_text = pytesseract.image_to_string(thresh)
opening_text = pytesseract.image_to_string(opening)
canny_text = pytesseract.image_to_string(canny)

# Print the extracted text
print("Text extracted from grayscale image:\n", gray_text)
print("\nText extracted from thresholded image:\n", thresh_text)
print("\nText extracted from image after opening:\n", opening_text)
print("\nText extracted from image after Canny edge detection:\n", canny_text)

# Plot the output of each step
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Convert the images to RGB format for correct display in matplotlib
axs[0, 0].imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Grayscale')

axs[0, 1].imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Thresholding')

axs[1, 0].imshow(cv2.cvtColor(opening, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Opening')

axs[1, 1].imshow(cv2.cvtColor(canny, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Canny Edge Detection')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()