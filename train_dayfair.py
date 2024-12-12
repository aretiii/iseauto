import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

epochs = 30
patience = 0
batch = 16
lr0 = 0.001
imgsz = 640
optimizer = 'AdamW'

folder_name = f"{epochs}_{patience}_{batch}_{lr0}_{imgsz}_{optimizer}"
base_path = "/home/areti/iseauto/training/runs"
output_dir = os.path.join(base_path, folder_name)
os.makedirs(output_dir, exist_ok=True)

train_log_file = os.path.join(output_dir, 'training_metrics.txt')
final_metrics_file = os.path.join(output_dir, 'final_metrics.txt')

model = YOLO('yolov8m.pt')
results = model.train(
    data='/home/areti/aretii/iseauto/pythonProject/training_yolov8/config.yaml',
    epochs=epochs,
    patience=patience,
    batch=batch,
    lr0=lr0,
    imgsz=imgsz,
    project=output_dir,
    optimizer=optimizer,
    name="train_run"
)

path_best_weights = os.path.join(output_dir, "train_run", "weights", "best.pt")
model = YOLO(path_best_weights)

train_metrics = model.val(split='train')
val_metrics = model.val(split='val')

with open(final_metrics_file, 'w') as f:
    f.write(f"Training Mean Average Precision at 95: {train_metrics.box.map}\n")
    f.write(f"Training Mean Average Precision at 50: {train_metrics.box.map50}\n")
    f.write(f"Training Mean Average Precision at 70: {train_metrics.box.map75}\n")
    f.write(f"Validation Mean Average Precision at 95: {val_metrics.box.map}\n")
    f.write(f"Validation Mean Average Precision at 50: {val_metrics.box.map50}\n")
    f.write(f"Validation Mean Average Precision at 70: {val_metrics.box.map75}\n")

print(f"Training Mean Average Precision at 95: {train_metrics.box.map}")
print(f"Validation Mean Average Precision at 95: {val_metrics.box.map}")

test_img_dir = "/home/areti/aretii/iseauto/pythonProject/training_yolov8/images/test"
prediction_dir = os.path.join(output_dir, "predictions")
os.makedirs(prediction_dir, exist_ok=True)

with torch.no_grad():
    results = model.predict(source=test_img_dir, conf=0.50, iou=0.75)

for result in results:
    if len(result.boxes.xyxy):
        name = result.path.split("/")[-1].split(".")[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        label_file_path = os.path.join(prediction_dir, name + ".txt")
        with open(label_file_path, "w") as f:
            for score, box in zip(scores, boxes):
                text = f"{score:0.4f} " + " ".join(box.astype(str))
                f.write(text)
                f.write("\n")