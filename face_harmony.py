import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define facial landmark indices for key features
LANDMARKS = {
    'left_eye_outer': 263,
    'left_eye_inner': 362,
    'right_eye_outer': 33,
    'right_eye_inner': 133,
    'nose_tip': 1,
    'mouth_left': 291,
    'mouth_right': 61,
    'chin': 152
}

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert image and process with MediaPipe
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    
    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract key facial landmarks
            left_eye_outer = face_landmarks.landmark[LANDMARKS['left_eye_outer']]
            left_eye_inner = face_landmarks.landmark[LANDMARKS['left_eye_inner']]
            right_eye_outer = face_landmarks.landmark[LANDMARKS['right_eye_outer']]
            right_eye_inner = face_landmarks.landmark[LANDMARKS['right_eye_inner']]
            nose_tip = face_landmarks.landmark[LANDMARKS['nose_tip']]
            mouth_left = face_landmarks.landmark[LANDMARKS['mouth_left']]
            mouth_right = face_landmarks.landmark[LANDMARKS['mouth_right']]
            chin = face_landmarks.landmark[LANDMARKS['chin']]
            
            # Convert normalized coordinates to pixel values
            h, w, _ = image.shape
            left_eye_outer_x = int(left_eye_outer.x * w)
            left_eye_outer_y = int(left_eye_outer.y * h)
            left_eye_inner_x = int(left_eye_inner.x * w)
            left_eye_inner_y = int(left_eye_inner.y * h)
            right_eye_outer_x = int(right_eye_outer.x * w)
            right_eye_outer_y = int(right_eye_outer.y * h)
            right_eye_inner_x = int(right_eye_inner.x * w)
            right_eye_inner_y = int(right_eye_inner.y * h)
            nose_tip_x = int(nose_tip.x * w)
            nose_tip_y = int(nose_tip.y * h)
            mouth_left_x = int(mouth_left.x * w)
            mouth_left_y = int(mouth_left.y * h)
            mouth_right_x = int(mouth_right.x * w)
            mouth_right_y = int(mouth_right.y * h)
            chin_x = int(chin.x * w)
            chin_y = int(chin.y * h)
            
            # Calculate facial proportions
            eye_distance = math.sqrt((right_eye_outer_x - left_eye_outer_x)**2 + (right_eye_outer_y - left_eye_outer_y)**2)
            nose_mouth_distance = math.sqrt((mouth_left_x - nose_tip_x)**2 + (mouth_left_y - nose_tip_y)**2)
            mouth_chin_distance = math.sqrt((chin_x - mouth_left_x)**2 + (chin_y - mouth_left_y)**2)
            
            # Calculate harmony score based on Golden Ratio
            harmony_score = 0
            if eye_distance > 0 and nose_mouth_distance > 0 and mouth_chin_distance > 0:
                ratio1 = nose_mouth_distance / eye_distance
                ratio2 = mouth_chin_distance / nose_mouth_distance
                if abs(ratio1 - 1.618) < 0.2 and abs(ratio2 - 1.618) < 0.2:
                    harmony_score = 9
                elif abs(ratio1 - 1.618) < 0.4 and abs(ratio2 - 1.618) < 0.4:
                    harmony_score = 7
                else:
                    harmony_score = 5
            
            # Draw landmarks and display harmony score
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
            cv2.putText(image, f"Facial Harmony Score: {harmony_score}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Facial Harmony Rating', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
