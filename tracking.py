import cv2
import time

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Allow the camera to warm up
time.sleep(2)

# Read the first frame from the webcam
ret, frame = cap.read()

# Select a region of interest (ROI) for the object to track
bbox = cv2.selectROI("Select Object", frame, False)
cv2.destroyWindow("Select Object")

# Initialize the tracker (using CSRT for better accuracy)
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

# Display the selected object for 10 seconds
start_time = time.time()
while int(time.time() - start_time) < 10:
    ret, frame = cap.read()
    if not ret:
        break
    # Draw the bounding box around the selected object
    (x, y, w, h) = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Start tracking after 10 seconds
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker and get the new position of the object
    success, bbox = tracker.update(frame)

    # Draw the bounding box on the frame
    if success:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with tracking
    cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
