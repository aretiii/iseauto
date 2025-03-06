import os
import rclpy
import rosbag2_py
import numpy as np
import open3d as o3d
import cv2
import struct
import time
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge


def extract_rosbag_data():
    rclpy.init()

    # Paths
    bag_path = "/home/autolab/Documents/RosBag_Areti_28Jan/rosbag2_2025_02_03-12_42_33"
    output_dir = "/home/autolab/Documents/PythonProject/extracted_data"

    # ROS Topics
    image_topic = "/sensing/camera/traffic_light/flir_camera/image_raw"
    lidar_topic = "/sensing/lidar/top/pointcloud_raw_ex"

    # Create output directories
    os.makedirs(f"{output_dir}/png", exist_ok=True)
    os.makedirs(f"{output_dir}/pcd", exist_ok=True)

    # Open the ROS bag
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    bridge = CvBridge()

    # Initialize counters
    image_counter = 0
    lidar_counter = 0

    while reader.has_next():
        (topic, data, t) = reader.read_next()

        if topic == image_topic:
            img_msg = deserialize_message(data, Image)
            cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")

            img_filename = os.path.join(output_dir, "png", f"{image_counter:06d}.png")  # 6-digit counter
            cv2.imwrite(img_filename, cv_image)
            print(f"Saved image: {img_filename}")
            image_counter += 1  # Increment counter

        elif topic == lidar_topic:
            pcd_msg = deserialize_message(data, PointCloud2)

            # Convert PointCloud2 message to numpy array
            points = []
            for i in range(0, len(pcd_msg.data), pcd_msg.point_step):
                x, y, z = struct.unpack_from('fff', pcd_msg.data, i)
                points.append([x, y, z])

            # Save as .pcd file
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd_filename = os.path.join(output_dir, "pcd", f"{lidar_counter:06d}.pcd")  # 6-digit counter
            o3d.io.write_point_cloud(pcd_filename, pcd)
            print(f"Saved LiDAR point cloud: {pcd_filename}")
            lidar_counter += 1  # Increment counter

        time.sleep(0.2)  # Slight delay to prevent excessive processing

    print("Extraction completed!")
    rclpy.shutdown()


if __name__ == "__main__":
    extract_rosbag_data()