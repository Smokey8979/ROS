import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define finger tip landmarks (MediaPipe hand landmarks)
FINGER_TIPS = {
    'thumb': 4,
    'index': 8,
    'middle': 12,
    'ring': 16,
    'pinky': 20
}

# Calibration variables
calibrated = False
pixel_distance = None
real_distance = 10  # cm, length of the calibration object
conversion_factor = None

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert image and process with MediaPipe
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark coordinates for thumb and index
            thumb_tip = hand_landmarks.landmark[FINGER_TIPS['thumb']]
            index_tip = hand_landmarks.landmark[FINGER_TIPS['index']]
            
            # Convert normalized coordinates to pixel values
            h, w, _ = image.shape
            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)
            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)
            
            # Calculate Euclidean distance in pixels
            pixel_distance = math.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
            
            # Draw landmarks and distance line
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Draw line between thumb and index
            cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (0,255,0), 2)
            
            if not calibrated:
                cv2.putText(image, "Place calibration object (10 cm) and press 'c' to calibrate", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                # Calculate distance in cm using conversion factor
                cm_distance = pixel_distance * conversion_factor
                
                # Display distance
                cv2.putText(image, f"Distance: {round(cm_distance, 2)} cm", 
                           ((thumb_x + index_x)//2, (thumb_y + index_y)//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow('Finger Distance Measurement', image)
    
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('c') and not calibrated:
        # Calibrate using the current pixel distance
        conversion_factor = real_distance / pixel_distance
        calibrated = True

cap.release()
cv2.destroyAllWindows()
