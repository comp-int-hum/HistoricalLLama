import matplotlib.pyplot as plt
import keras_ocr
import argparse
import glob 
import json
import fitz
from PIL import Image
import io

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", dest='input_file', required = True, help="input_file_folder")
parser.add_argument("--output_file", dest= "output_file", required = True,  help = "output_ocr_text")



args = parser.parse_args()
pipeline = keras_ocr.pipeline.Pipeline()
out_text = {}

pathway = args.input_file

combined_text = []
doc = fitz.open(pathway)
for i in range(len(doc)):
        page = doc[i]
        pixmap = page.get_pixmap()
        png_bytes = pixmap.tobytes()
        image = io.BytesIO(png_bytes)
        read = keras_ocr.tools.read(image)
        prediction_groups = pipeline.recognize([read])

        for pre_diction_group in prediction_groups: 
            recognized_text = [word for word, _ in pre_diction_group]
            recognized_text = " ".join(recognized_text)
            combined_text.append(recognized_text)
out_text[pathway] = combined_text


with open(args.output_file, "w") as json_out:
    json_out.write(json.dumps(out_text, indent = 4))
        
#print(len(images))
# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
#for entry in images:
 #   prediction_group = pipeline.recognize(entry[1])
  #  recognized_text = [word for word, _ in prediction_group]
  #  out_text[entry[0]] = recognized_text 
#print(out_text)
    
#$# Plot the predictions
#fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
#for ax, image, predictions in zip(axs, images, prediction_groups):
 #   keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

    
    


