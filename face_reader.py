import cv2
import sys
import numpy as np
import mss
import time
from deepface import DeepFace
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPalette, QColor, QFont

# Load a pre-trained Haar Cascade classifier for face detection
# Make sure you have the 'haarcascade_frontalface_default.xml' file
# You might need to specify the full path to the cascades directory
# Example path structure if installed via pip: 
# sys.prefix + '/lib/pythonX.Y/site-packages/cv2/data/haarcascade_frontalface_default.xml'
# Or download it from OpenCV's GitHub repository: 
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
try:
    # Construct the path relative to the cv2 package installation
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError(f"Could not load face cascade classifier from {cascade_path}")
except Exception as e:
    print(f"Error loading cascade file: {e}", file=sys.stderr)
    print("Please ensure OpenCV is installed correctly and the cascade file exists.", file=sys.stderr)
    print("You may need to download 'haarcascade_frontalface_default.xml' and place it in your project directory, then update the path here.", file=sys.stderr)
    sys.exit(1)

# --- Screen Capture Setup ---
# Define the screen area to capture (e.g., primary monitor)
# You can adjust 'top', 'left', 'width', 'height' for a specific region
# Or use monitor=N for a specific monitor index (usually 1 is primary)
monitor_definition = {"top": 0, "left": 0, "width": 1920, "height": 1080} # Adjust width/height to your primary monitor resolution if needed
sct = mss.mss()
# --- End Screen Capture Setup ---

# Start video capture from the default webcam (usually index 0)
# video_capture = cv2.VideoCapture(0) # No longer using webcam

# if not video_capture.isOpened():
#     print("Error: Could not open webcam.", file=sys.stderr)
#     sys.exit(1)

print("Starting screen capture analysis. Press 'q' to quit.")

# --- PyQt Application Setup ---
app = QApplication(sys.argv)

# Get screen geometry
screen = app.primaryScreen()
screen_geometry = screen.geometry()
screen_width = screen_geometry.width()
screen_height = screen_geometry.height()

# Create a widget to hold the label
overlay_widget = QWidget()
overlay_widget.setWindowFlags(
    Qt.WindowType.WindowStaysOnTopHint |
    Qt.WindowType.FramelessWindowHint
)
# Make background transparent (or semi-transparent)
# overlay_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground) # Commenting this out
# For better visibility, let's start with a semi-transparent black background
overlay_widget.setStyleSheet("background-color: rgba(0, 0, 0, 150); border-radius: 5px;") # Re-enabled stylesheet background

# Create the label for emotion text
emotion_label = QLabel("Emotion: Waiting...", overlay_widget)
emotion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
emotion_label.setFont(QFont("Arial", 16)) # Adjust font size as needed
palette = emotion_label.palette()
palette.setColor(QPalette.ColorRole.WindowText, QColor("white")) # Set text color
emotion_label.setPalette(palette)

# Position and size the widget/label at the bottom center
label_width = 300
label_height = 50
label_x = (screen_width - label_width) // 2
label_y = screen_height - label_height - 20 # Position 20px from bottom

overlay_widget.setGeometry(label_x, label_y, label_width, label_height)
emotion_label.setGeometry(0, 0, label_width, label_height) # Label fills the widget

overlay_widget.show()
# --- End PyQt Application Setup ---

# --- Global variable for dominant emotion ---
global_dominant_emotion = "Waiting..."

# --- Analysis Function (to run periodically) ---
def analyze_screen():
    global global_dominant_emotion
    detected_emotion = "Neutral" # Default if no face found
    face_found = False
    
    try:
        # Capture screen
        sct_img = sct.grab(monitor_definition)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50) # Slightly larger minSize might improve performance
        )

        # Analyze the *first* detected face for simplicity
        if len(faces) > 0:
            face_found = True
            (x, y, w, h) = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                if isinstance(result, list) and len(result) > 0:
                    detected_emotion = result[0]['dominant_emotion']
                elif isinstance(result, dict):
                     detected_emotion = result.get('dominant_emotion', 'N/A')
                else:
                    detected_emotion = 'N/A'
            except ValueError as ve:
                # Handle case where face ROI is too small or invalid for DeepFace
                # print(f"DeepFace error: {ve}")
                detected_emotion = "Invalid ROI" # Or keep previous emotion
            except Exception as e:
                # print(f"Error analyzing face: {e}")
                detected_emotion = "Error" # Or keep previous emotion
        else:
            # No face detected, keep emotion as Neutral or previous?
            detected_emotion = "No Face"

    except mss.ScreenShotError as ex:
        print(f"Error capturing screen: {ex}", file=sys.stderr)
        time.sleep(0.5)
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}", file=sys.stderr)
        # Potentially stop the timer or handle differently
        detected_emotion = "Capture Error"

    global_dominant_emotion = detected_emotion
    emotion_label.setText(f"Emotion: {global_dominant_emotion}")

# --- Timer to run analysis periodically ---
# Analyze every 500ms (adjust as needed for performance)
timer = QTimer()
timer.timeout.connect(analyze_screen)
timer.start(500) 

print("Starting screen analysis for emotion overlay. Press Ctrl+C in terminal to quit.")

# Start the PyQt event loop
sys.exit(app.exec())
