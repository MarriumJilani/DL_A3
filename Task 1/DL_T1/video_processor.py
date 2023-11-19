import cv2
import numpy as np
from sort import Sort
import time

class VideoProcessor:
    def __init__(self, video_path, yolo_detector):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.roi = None
        self.tracker = Sort()
        self.total_cars_count = 0
        self.total_motorcycles_count = 0
        self.total_trucks_count = 0
        self.yolo_detector = yolo_detector
        self.prev_frame_objects = None
        self.prev_frame_time = 0
        self.prev_frame_objects = []

    def process_video(self):
        _, first_frame = self.cap.read()
        counted_ids = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.roi is not None:
                frame_roi = frame[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]]
            else:
                frame_roi = frame

            # Object detection using YOLO
            detected_objects = self.yolo_detector.detect_objects(frame_roi)

            # Tracking using SORT
            if detected_objects:
                tracked_objects = self.tracker.update(
                    np.array([[obj['box'][0], obj['box'][1], obj['box'][0] + obj['box'][2], obj['box'][1] + obj['box'][3],
                                obj['class_id']] for obj in detected_objects]))
                speed=self.estimate_speed(tracked_objects, self.prev_frame_objects, frame)  # Pass the previous frame objects
                self.display_objects(frame, tracked_objects, detected_objects, counted_ids,speed)  # Pass the list of counted IDs

            cv2.imshow('Processed Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Update previous frame objects
            self.prev_frame_objects = tracked_objects
            self.prev_frame_time = time.time()  # Update the previous frame time

        self.cap.release()
        cv2.destroyAllWindows()

    def estimate_speed(self, tracked_objects, prev_frame_objects, frame):
        frame_time = time.time()
        frame_rate = 1 / (frame_time - self.prev_frame_time) if self.prev_frame_time != 0 else 0
        pixels_per_foot = 8  # Adjust this value based on your scale information
        meters_per_pixel = 0.000264583  # Adjust this value based on your scale information

        speeds = []  # List to store calculated speeds for each object

        for obj, prev_obj in zip(tracked_objects, prev_frame_objects):
            # Extract relevant information
            bbox = obj[:4]
            prev_bbox = prev_obj[:4]

            # Calculate speed
            speed = self.calculate_speed(bbox, prev_bbox, frame_rate, pixels_per_foot, meters_per_pixel)
            speeds.append(speed)

        return speeds

    def calculate_speed(self, bbox, prev_bbox, frame_rate, pixels_per_foot, meters_per_pixel):
        # Calculate distance traveled in pixels
        distance_pixels = np.sqrt((bbox[0] - prev_bbox[0]) ** 2 + (bbox[1] - prev_bbox[1]) ** 2)

        distance_meters = distance_pixels * meters_per_pixel

        speed_meters_per_second = distance_meters * frame_rate

        speed_kmh = speed_meters_per_second * 3.6

        return speed_kmh

    def display_objects(self, frame, tracked_objects, detected_objects, counted_ids, speeds):
        cars_count = 0
        motorcycles_count = 0
        trucks_count = 0

        current_frame_ids = []  # List to store object IDs detected in the current frame

        
        for detection, speed in zip(detected_objects, speeds):
            
            class_id = int(detection['class_id'])
            confidence = detection['confidence']
            print(f"class id:{class_id}")
            # Check if the detected object is a car, motorcycle, or truck based on class_id
            if class_id == 2 and confidence > 0.7:  
                cars_count += 1
                #print(f"TNT!!! {cars_count}")
            elif class_id == 3 and confidence > 0.5:  
                motorcycles_count += 1
            elif class_id == 7 and confidence > 0.5: 
                trucks_count += 1

        # Display tracked objects
        for obj, speed in zip(tracked_objects, speeds):
            bbox = obj[:4]
            object_id = int(obj[4])
            #class_id = int(obj[5]) if len(obj) > 5 else None  # Check if class ID is available
            
            # Adjust bounding box coordinates based on ROI
            bbox_adjusted = [
                int(bbox[0]) + self.roi[0],
                int(bbox[1]) + self.roi[1],
                int(bbox[2]) + self.roi[0],
                int(bbox[3]) + self.roi[1]
            ]

            # Draw adjusted bounding box
            cv2.rectangle(frame, (bbox_adjusted[0], bbox_adjusted[1]), (bbox_adjusted[2], bbox_adjusted[3]), (0, 255, 0), 2)

            # Display object ID
            cv2.putText(frame, f'ID: {object_id}', (bbox_adjusted[0], bbox_adjusted[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            # Display speed if available
            cv2.putText(frame, f'Speed: {speed:.2f} km/h',
                        (int((bbox_adjusted[0] + bbox_adjusted[2]) / 2) - 30, int((bbox_adjusted[1] + bbox_adjusted[3]) / 2) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            


            # Check if the object ID has already been counted in previous frames
            if object_id not in counted_ids:
                current_frame_ids.append(object_id)  # Add to the list for the current frame
                #print(f"here!!! {counted_ids}")
                # Increment the corresponding count based on class ID
                if class_id == 2:  # Car class
                    cars_count += 1
                    #print(f"OMG!!! {cars_count}")
                    self.total_cars_count += 1  # Update total count
                   # print(f"LOLOL!!! {self.total_cars_count}")
                elif class_id == 3:  # Motorcycle class
                    motorcycles_count += 1
                    self.total_motorcycles_count += 1  # Update total count
                elif class_id == 7:  # Truck class
                    trucks_count += 1
                    self.total_trucks_count += 1  # Update total count

        # Display total count on the frame
        # cv2.putText(frame, f'Cars: {cars_count}, Motorcycles: {motorcycles_count}, Trucks: {trucks_count}',
        #             (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Total Cars: {self.total_cars_count}',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f'Total Motorcycles: {self.total_motorcycles_count}',
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f'Total Trucks: {self.total_trucks_count}',
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update the counted_ids list with all the unique object IDs detected in the current frame
        counted_ids.extend(current_frame_ids)

        print(f"Total Cars: {self.total_cars_count}, Total Motorcycles: {self.total_motorcycles_count}, Total Trucks: {self.total_trucks_count}")
        print(f"Cars: {cars_count}, Motorcycles: {motorcycles_count}, Trucks: {trucks_count}")

