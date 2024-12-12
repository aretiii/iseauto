import os
import cv2
import numpy as np
from ipython_genutils.py3compat import getcwd

def convert(size, x_center, y_center, width, height):
    img_w, img_h = size
    xmin = int((x_center - width / 2) * img_w)
    xmax = int((x_center + width / 2) * img_w)
    ymin = int((y_center - height / 2) * img_h)
    ymax = int((y_center + height / 2) * img_h)
    return xmin, ymin, xmax, ymax

mypath = "/home/areti/data/iseauto_dataset_bbox/day_fair/"
outpath = "/home/areti/data/result1/"

image_name = "sq21_002862.jpg"
txt_name = "sqq21_002862.txt"

img_path = os.path.join(mypath, image_name)
text_path = os.path.join(mypath, txt_name)

img = cv2.imread(img_path)
if img is None:
    print(f"Image {image_name} is empty")
else:
    with open(text_path, "r") as txt_file:
        lines = txt_file.read().splitlines()
    for idx, line in enumerate(lines):
        value = line.split()
        cls = value[0]
        x = float(value[1])
        y = float(value[2])
        w = float(value[3])
        h = float(value[4])
        img_h, img_w = img.shape[:2]
        xmin, ymin, xmax, ymax = convert((img_w, img_h), x, y, w ,h)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        img_outpath = os.path.join(outpath, image_name)
        cv2.imwrite(img_outpath, img)