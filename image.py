import os
import cv2
import numpy as np

def convert(size, x_center, y_center, width, height):
    img_w, img_h = size
    xmin = int((x_center - width / 2) * img_w)
    xmax = int((x_center + width / 2) * img_w)
    ymin = int((y_center - height / 2) * img_h)
    ymax = int((y_center + height / 2) * img_h)
    return xmin, ymin, xmax, ymax

mypath = "/home/areti/data/iseauto_dataset_bbox/day_fair/"
outpath = "/home/areti/data/result/"

image_name = "sq11_000000.jpg"
txt_name = "sq11_000000.txt"

img_path = os.path.join(mypath, image_name)
txt_path = os.path.join(mypath, txt_name)

img = cv2.imread(img_path)
if img is None:
    print(f"Image {image_name} not found!")
else:
    img_h, img_w = img.shape[:2]

    with open(txt_path, "r") as txt_file:
        lines = txt_file.readlines()

    for line in lines:
        values = line.split()
        cls = values[0]
        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])

        xmin, ymin, xmax, ymax = convert((img_w, img_h), x_center, y_center, width, height)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        label = f"Class {cls}"
        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

img_outpath = os.path.join(outpath, image_name)
cv2.imwrite(img_outpath, img)
print(f"Processed and saved: {img_outpath}")