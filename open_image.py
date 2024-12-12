import cv2
image = cv2.imread("/home/areti/data/iseauto_dataset_bbox/day_fair/sq11_000000.jpg", cv2.IMREAD_COLOR)
if image is None:
    print("Failed to load image.")
else:
    print("Image loaded successfully.")
resize_image = cv2.resize(image, (480,320))
#cv2.imshow("image", image)
cv2.imshow("resize_image", resize_image)
save_path = "/home/areti/data/iseauto_dataset_bbox/resized_image.jpg"
cv2.imwrite(save_path, resize_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
