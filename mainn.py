import sys
import site
import numpy as np  # type: ignore
import cv2  # type: ignore
import time
import os

# Print executable and site packages (for debugging, if needed)
print(sys.executable)
print(site.getsitepackages())

prototxt = r"C:\Users\saranya\Downloads\797a3e7ee041ef88cd4d9e293eaacf9f-3d2765b625f1b090669a05d0b3e79b2907677e86\797a3e7ee041ef88cd4d9e293eaacf9f-3d2765b625f1b090669a05d0b3e79b2907677e86\MobileNetSSD_deploy.prototxt"
model = r"C:\Users\saranya\Downloads\archive (10)\MobileNetSSD_deploy.caffemodel"
confThresh = 0.2
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor", "mobile"]

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("Model Loaded")
print("Starting Camera Feed...")

# Initialize video capture
vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    # Read frame from video stream
    ret, frame = vs.read()
    if not ret:
        break

    # Resize the frame to make it smaller for faster processing
    frame = cv2.resize(frame, (1000, 600))  # Adjust resolution as needed
    (h, w) = frame.shape[:2]

    # Prepare the frame for model input
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Process detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confThresh:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Show the output frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and close windows
vs.release()
cv2.destroyAllWindows()
