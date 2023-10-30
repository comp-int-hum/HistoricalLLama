import json
import glob
import argparse
import nltk

nltk.download('words')
from nltk.corpus import words
all_the_words_list = words.words()

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", dest='input_file', help="input_file_folder", nargs = "+")
parser.add_argument("--output_file", dest= "output_file", help = "output_ocr_text")

args = parser.parse_args()
out_dict = {}
pathway = args.input_file
#print(pathway)
for path in pathway:
    print("this is the path")
    print(path)
    with open(path, 'r') as zip_file:
        path = path.split("from")
        path = path[1]
        print(path)
        dictionary = json.load(zip_file)
        for key, item in dictionary.items():
            print(key)
            print(item)
            if out_dict.get(key):
                counter = 0
                word_len = 0
                temp_list = out_dict[key]
                for page in item:
                    print("this is the page")
                    print(page)
                    page  = page.split()
                    word_len = word_len + len(page)
                    for word in page:
                        if word in all_the_words_list:
                            counter = counter + 1
                if counter > 0:
                    if word_len > 0:
                        percentage = counter / (word_len)
                else:
                    percentage = 0
                temp_list.append([path,percentage,item])
                out_dict[key] = temp_list
            else:
                temp_list = []
                counter = 0
                word_len = 0 
                for page in item:
                    print(page)
                    print("post-page")
                    page  = page.split()
                    print(page)
                    print(len(page))
                    word_len = word_len + len(page)
                    for word in page:
                        if word in all_the_words_list:
                            counter = counter + 1
                if counter > 0:
                    if word_len > 0:
                        percentage = counter / (word_len)
                else:
                    percentage = 0
                temp_list.append([path,percentage, item])
                out_dict[key] = temp_list
with open(args.output_file, "w") as out_file:
    out_file.write(json.dumps(out_dict, indent = 4))
            
