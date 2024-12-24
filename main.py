import cv2
import numpy as np
import math
from ultralytics import YOLO
from sort import Sort
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Car Counter')
    parser.add_argument('video_path', type=str, help='Path to the video')
    return parser.parse_args()

def init_model(model_path='yolov8l.pt'):
    return YOLO(model_path)

def create_mask(frame):
    height, width = frame.shape[:2]
    mask_coords = [(0, height // 2), (width, height)]
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, mask_coords[0], mask_coords[1], 255, -1)
    return mask, mask_coords

def detect_objects(model, frame, mask):
    class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
    "teddy bear", "hair drier", "toothbrush"
    ]   
    
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    results = model(masked_frame, stream=True)

    detections = np.empty((0, 5))
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = math.ceil(box.conf[0] * 100) / 100
            class_index = int(box.cls[0])

            if class_index < len(class_names) and class_names[class_index] == 'car' and confidence > 0.5:
                detection = np.array([[x1, y1, x2, y2, confidence]])
                detections = np.vstack((detections, detection))

    return detections

def track_objects(tracker, detections):
    return tracker.update(detections)

def count_cars(tracked_objects, line, total_counts):
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if line[0] < cx < line[2] and line[1] - 15 < cy < line[3] + 15:
            if obj_id not in total_counts:
                total_counts.append(obj_id)
    return total_counts

def draw_boxes(frame, detections, tracked_objects, total_counts, line, mask_coords):
    cv2.rectangle(frame, mask_coords[0], mask_coords[1], (0, 0, 255), 2)
    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

    for detection in detections:
        x1, y1, x2, y2, _ = map(int, detection)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.putText(frame, f'{obj_id}', (x1, max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), cv2.FILLED)

    cv2.putText(frame, f'Count: {len(total_counts)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def car_counter(video_path):
    cap = cv2.VideoCapture(video_path)
    model = init_model()
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    total_counts = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        mask, mask_coords = create_mask(frame)
        detections = detect_objects(model, frame, mask)
        tracked_objects = track_objects(tracker, detections)
        total_counts = count_cars(tracked_objects, line=[0, frame.shape[0] - 100, frame.shape[1], frame.shape[0] - 100], total_counts=total_counts)
        frame = draw_boxes(frame, detections, tracked_objects, total_counts, line=[0, frame.shape[0] - 100, frame.shape[1], frame.shape[0] - 100], mask_coords=mask_coords)

        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Output", 800, 600)
        cv2.imshow('Output', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    args = parse_args()
    car_counter(args.video_path)
