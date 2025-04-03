import torch
import cv2

# Load YOLOv5 pre-trained model (you can use 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Choose smaller model for real-time performance (yolov5s)

# Open the webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)

    # Render results on the frame (bounding boxes and labels)
    frame = results.render()[0]  # The rendered frame with bounding boxes

    # Show the frame with bounding boxes in a window
    cv2.imshow('Real-Time Object Detection', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
