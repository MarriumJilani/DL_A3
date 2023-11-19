import cv2
import tkinter as tk
from tkinter import filedialog
from threading import Thread
import numpy as np
from sort import Sort
from yolo_detector import YOLODetector
from video_processor import VideoProcessor

if __name__ == "__main__":
    yolo_config_path = 'yolov3.cfg'
    yolo_weights_path = 'yolov3.weights'
    yolo_labels_path = 'coco.names'

    yolo_detector = YOLODetector(yolo_config_path, yolo_weights_path, yolo_labels_path)

    root = tk.Tk()  # Create the Tkinter root instance
    root.title("Vehicle Counting and Speed Estimation")

    video_processor = VideoProcessor("", yolo_detector)

    def select_video():
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            video_processor.video_path = file_path
            print(file_path)
            video_processor.cap = cv2.VideoCapture(file_path)
            _, frame = video_processor.cap.read()
            video_processor.roi = cv2.selectROI(frame, False)
            video_processor.process_video()
        else:
            print("Not able to capture video")

    select_video_button = tk.Button(root, text="Select Video", command=select_video)
    select_video_button.pack()

    root.mainloop()
