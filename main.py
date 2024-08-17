import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Function to perform object detection
def detect_objects(frame, model):
    results = model(frame)
    detections = {}
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = result.names[box.cls[0].item()]
            confidence = box.conf[0].item()
            
            if label not in detections:
                detections[label] = []
            detections[label].append((x1, y1, x2, y2, confidence))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, detections

# Streamlit app interface
st.title("Real-Time Video Input with YOLOv8 Object Detection")

# Initialize webcam video stream
cap = cv2.VideoCapture(0)

# Dictionary to store detected objects
detected_objects = {}

stframe = st.empty()

# Process the video stream in real-time
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture video from webcam.")
        break

    # Perform object detection
    processed_frame, detections = detect_objects(frame, model)
    
    # Update detected objects
    for label, coords in detections.items():
        if label in detected_objects:
            detected_objects[label].extend(coords)
        else:
            detected_objects[label] = coords

    # Display the processed video
    stframe.image(processed_frame, channels="BGR")

    # To control the frame rate (optional)
    time.sleep(0.1)

cap.release()

# Display detected objects
st.write("Detected Objects:")
st.json(detected_objects)
