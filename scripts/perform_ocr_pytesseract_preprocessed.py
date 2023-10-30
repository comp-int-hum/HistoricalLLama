import os
import subprocess
from PIL import Image
import glob
import io
import pytesseract
from PyPDF2 import PdfReader
import argparse
import json
import fitz
import cv2
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument("--input_file", dest='input_file', help="input_file_folder")
parser.add_argument("--output_file", dest= "output_file", help = "output_ocr_text")


args = parser.parse_args()




def preprocess_image(page):
    # Convert the PDF page to a pixmap
    matrix = fitz.Matrix(2, 2)  # Adjust the numbers for your specific needs                                                                                                                                      
    pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)

    pixmap = page.get_pixmap()
    
    # Convert the pixmap to bytes
    png_bytes = pixmap.tobytes()
    
    # Create a PIL Image from the bytes
    image = Image.open(io.BytesIO(png_bytes))
    
    # Convert the PIL Image to a NumPy array for OpenCV
    image_np = np.array(image)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Noise removal using median blur
    denoised_image = cv2.medianBlur(gray_image, 3)
    
    # Binary thresholding
    _, binary_image = cv2.threshold(denoised_image, 128, 255, cv2.THRESH_BINARY)
    
    # Convert the NumPy array back to a PIL Image
    processed_image = Image.fromarray(binary_image)
    
    return processed_image



def make_ocr(the_file):
    text_file = []
    for i in range(len(the_file)):
        page = the_file[i]
        image = preprocess_image(page)
        
        #there are also definitely modes to further test/explore here, including page/column segmentation mode                                                                                                       
        text = pytesseract.image_to_string(image)
        text = text.replace('\n', ' ')
        text_file.append(text)
        " ".join(text_file)
        return text_file


text_list = {}


pathway = args.input_file



with open (pathway, 'r') as zip_file:
    doc = fitz.open(zip_file)
    out_text = make_ocr(doc)
    text_list[pathway] = out_text

with open (args.output_file, "w") as out_file:
    out_file.write(json.dumps(text_list, indent = 4))
