This project is a real-time object detection application that uses a standard webcam to identify and classify objects in a live video feed. It is built with Python and leverages the power of the OpenCV library and the pre-trained YOLOv3 (You Only Look Once) model.

Features
Real-Time Detection: Identifies objects from a live webcam feed.

YOLOv3 Model: Utilizes the powerful and popular YOLOv3 model for fast and accurate detections.

80 Object Classes: Can detect and classify up to 80 different types of objects based on the COCO dataset.

Confidence Scoring: Filters out weak detections by only considering objects with a confidence score greater than 50%.

Non-Max Suppression: Cleans up the output by removing redundant, overlapping bounding boxes for the same object, ensuring a single, accurate detection.

Visual Annotation: Draws green bounding boxes around detected objects and labels them with the class name and confidence percentage.

How It Works:
The script captures frames from the webcam and converts each frame into a blob. This blob is then fed into the YOLOv3 neural network. The network outputs potential object detections, which are then filtered based on confidence scores. Non-Max Suppression is applied to select the best bounding box for each detected object. Finally, the resulting frame with the detections is displayed on the screen.

Prerequisites
Before you begin, ensure you have the following installed:

Python 3

OpenCV library (opencv-python)

NumPy library

You can install the required Python libraries using pip:

Bash
pip install opencv-python numpy
Required Files
You also need the pre-trained YOLOv3 model files in the same directory as the script. You can download them from the official YOLO website or other sources.

yolov3.weights: The pre-trained weights of the YOLOv3 model.

yolov3.cfg: The configuration file for the YOLOv3 model.

coco.names: A text file containing the names of the 80 classes the model can detect.

How to Run
Clone this repository or place all the required files (object_detector.py, yolov3.weights, yolov3.cfg, coco.names) in the same folder.

Open your terminal or command prompt and navigate to the project directory.

Run the script with the following command:

Bash
python object_detector.py
A window titled "Object Detection" will appear, showing your webcam feed with objects being detected in real-time.

To stop the application, press the 'q' key on your keyboard.
