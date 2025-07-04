# Step 1: Importing necessary libraries
import cv2
import numpy as np

# Model and Class loading
# step 2: Load the YOLO Model
# We load the network using its configuration and weights files.
print("Loading YOLO Model...")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Step 3: Load the class names
# This file contains the names of the 80 ojects the model can detect.
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Steo 4: Get the output layer names
# These are the layers of the network that will give us the final object detections.
layer_names = net.getLayerNames()
output_layers = [layer_names[i -1] for i in net.getUnconnectedOutLayers()]
print("YOLO Model loaded successfully.")

# WEBCAM Initialization
# Step 5: Initialize the webcam
# cv2.VideoCapture(0) accesses the default webcam.
print("Starting webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open Webcam.")
    exit()

# Real time detection loop
# Step 6: Start the main lopp to read frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare image for yolo
    # Step 7: Create a 'blob' from the image
    # A blob is the format the YOLO model expects.
    # (416, 416) is the size of the image YOLO was trained on.
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Step 8: Set the input for the network and get the output
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process Detections
    class_ids = []
    confidences = []
    boxes = []

    # Step 9: Lopp through the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Fliter out weak detections by ensuring the confidence is high enough 
            if confidence > 0.5:
                # Object Detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # rectange coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Step 10; Apply Non-max spression
    # this gets rid of redundant, overlapping boxes for the same object.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw Boxes and labels
    # Step 11: Draw the final boxes on the frame
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_label = f"{confidences[i]*100:.2f}%"
            color = (0, 255, 0) # Green color for the box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence_label}", (x, y + 30), font, 2, color, 3)

    # Display the frame 
    # Step 12: Show the frame in a window
    cv2.imshow("Object Detection", frame)

    # Step 13: Wait for the 'q' key to be pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup 
#Step 14: Release the webcam and close all windows 
print("Closing application...")
cap.release()
cv2.destroyAllWindows()