import cv2
import sys
import numpy as np
import mss
import time
import argparse
import google.generativeai as genai
import os
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPalette, QColor, QFont, QFontMetrics
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
import io

# --- Load .env file ---
load_dotenv()
# ---

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(description='Screen Face Emotion Analyzer')
parser.add_argument('--debug', action='store_true', 
                    help='Show the detected face ROI in a separate window.')
args = parser.parse_args()
# --- End Argument Parser Setup ---

# --- Google AI Setup ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    # Using a model that supports vision input
    # Adjust model name if needed (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro')
    genai_model = genai.GenerativeModel('gemini-1.5-flash') 
    print("Google Generative AI configured.")
except Exception as e:
    print(f"Error configuring Google Generative AI: {e}", file=sys.stderr)
    print("Please ensure the GOOGLE_API_KEY environment variable is set correctly.", file=sys.stderr)
    sys.exit(1)
# --- End Google AI Setup ---

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
sct = mss.mss()
# Use monitor 0 to capture the entire virtual screen (all monitors combined)
monitor_definition = sct.monitors[0]
print(f"Capturing entire virtual screen area: {monitor_definition}")
# --- End Screen Capture Setup ---

# --- Audio Capture Setup ---
SAMPLE_RATE = 16000 # Sample rate in Hz (common for speech models)
AUDIO_DURATION_SECONDS = 1.5 # Capture duration (adjust as needed)
AUDIO_CHANNELS = 1 # Mono audio
print(f"Audio capture setup: {AUDIO_DURATION_SECONDS}s @ {SAMPLE_RATE}Hz, {AUDIO_CHANNELS} channel(s)")
# Optional: Check available devices
# print(sd.query_devices())
# --- End Audio Capture Setup ---

# Start video capture from the default webcam (usually index 0)
# video_capture = cv2.VideoCapture(0) # No longer using webcam

# if not video_capture.isOpened():
#     print("Error: Could not open webcam.", file=sys.stderr)
#     sys.exit(1)

print("Starting screen capture analysis. Press 'q' to quit.")

# --- PyQt Application Setup ---
app = QApplication(sys.argv)

screen = app.primaryScreen()
screen_geometry = screen.geometry()
screen_width = screen_geometry.width()
screen_height = screen_geometry.height()

overlay_widget = QWidget()
overlay_widget.setWindowFlags(
    Qt.WindowType.WindowStaysOnTopHint |
    Qt.WindowType.FramelessWindowHint
)
overlay_widget.setStyleSheet("background-color: rgba(0, 0, 0, 150); border-radius: 5px;") 

emotion_label = QLabel("Waiting...", overlay_widget)
emotion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
emotion_label.setFont(QFont("Arial", 16))
palette = emotion_label.palette()
palette.setColor(QPalette.ColorRole.WindowText, QColor("white"))
emotion_label.setPalette(palette)
emotion_label.setWordWrap(True) # <-- Enable word wrap

# Initial Position and Size (will be adjusted dynamically)
initial_width = 200 
initial_height = 50 # Initial height, will change
initial_x = (screen_width - initial_width) // 2
initial_y = screen_height - initial_height - 20

overlay_widget.setGeometry(initial_x, initial_y, initial_width, initial_height)
emotion_label.setGeometry(0, 0, initial_width, initial_height) 

overlay_widget.show()
# --- End PyQt Application Setup ---

# --- OpenCV Window Setup for ROI (Conditional) ---
roi_window_name = 'Detected Face ROI'
if args.debug:
    print("Debug mode enabled: Showing detected face ROI window.")
    cv2.namedWindow(roi_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(roi_window_name, 150, 150) 
    cv2.moveWindow(roi_window_name, 10, 10) 
# --- End OpenCV Window Setup ---

# --- Global variable for dominant emotion ---
global_dominant_emotion = "Waiting..."

# --- Analysis Function (to run periodically) ---
def analyze_screen():
    global global_dominant_emotion
    detected_text = "Neutral" 
    face_found = False
    largest_face_roi = None 
    audio_data = None
    
    try:
        # --- Capture Screen and Audio Concurrently (simple approach) ---
        # Note: For better sync, threading might be needed, but let's keep it simple for POC
        
        # Capture screen
        sct_img = sct.grab(monitor_definition)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Capture audio
        try:
            # print("[Debug] Recording audio...") # Uncomment for debug
            audio_data = sd.rec(int(AUDIO_DURATION_SECONDS * SAMPLE_RATE), 
                                samplerate=SAMPLE_RATE, 
                                channels=AUDIO_CHANNELS, 
                                dtype='float32') # Use float32, soundfile handles conversion
            sd.wait() # Wait for recording to complete
            # print("[Debug] Audio recording complete.") # Uncomment for debug
        except Exception as audio_err:
            print(f"[Error] Failed to record audio: {audio_err}")
            audio_data = None # Ensure audio_data is None if capture fails
        # --- End Capture ---

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50) 
        )
        
        # --- Find Largest Face ---
        largest_area = 0
        best_face_coords = None

        for (x, y, w, h) in faces:
            area = w * h
            if area > largest_area:
                largest_area = area
                best_face_coords = (x, y, w, h)
        # --- End Find Largest Face ---

        # Analyze the largest detected face & audio
        if best_face_coords is not None:
            face_found = True
            x, y, w, h = best_face_coords
            largest_face_roi = frame[y:y+h, x:x+w]

            if args.debug and largest_face_roi.size > 0:
                cv2.imshow(roi_window_name, largest_face_roi)

            # --- Prepare Image and Audio for Gemini API --- 
            if largest_face_roi.size > 0 and audio_data is not None:
                try:
                    # 1. Prepare Image Part
                    is_success, buffer = cv2.imencode(".png", largest_face_roi)
                    if not is_success: raise ValueError("Failed to encode face ROI.")
                    image_bytes = buffer.tobytes()
                    image_part = {"mime_type": "image/png", "data": image_bytes}

                    # 2. Prepare Audio Part
                    # Encode audio to WAV in memory
                    wav_buffer = io.BytesIO()
                    sf.write(wav_buffer, audio_data, SAMPLE_RATE, format='WAV', subtype='PCM_16')
                    wav_buffer.seek(0)
                    audio_bytes = wav_buffer.read()
                    audio_part = {"mime_type": "audio/wav", "data": audio_bytes}
                    
                    # 3. Define the prompt (considering both modalities)
                    # *** WARNING: Still highly unreliable for deception detection. ***
                    prompt = f"Analyze the visual cues (facial expression, gaze) in the image AND the tone of voice in the audio. Based ONLY on these inputs, assess cues commonly *misinterpreted* as deception. State your confidence (High/Medium/Low). Answer very concisely."
                    
                    # 4. Prepare contents list
                    contents = [image_part, audio_part, prompt]
                    
                    # 5. Make API Call
                    response = genai_model.generate_content(contents)
                    detected_text = response.text.strip() if response.text else "API No Text"
                    
                except Exception as e:
                    print(f"[Error] Gemini API call failed: {e}")
                    detected_text = "API Error"
            elif largest_face_roi.size <= 0:
                detected_text = "Invalid ROI"
            else: # audio_data is None
                detected_text = "Audio Error"
            # --- End API Analysis --- 

        else:
            detected_text = "No Face"
            if args.debug:
                clear_frame = np.zeros((150, 150, 3), dtype=np.uint8) 
                cv2.imshow(roi_window_name, clear_frame)

    except mss.ScreenShotError as ex:
        print(f"Error capturing screen: {ex}", file=sys.stderr)
        detected_text = "Capture Error"
        time.sleep(0.5)
    except KeyboardInterrupt:
        print("Ctrl+C detected, exiting...")
        cv2.destroyAllWindows() # Clean up OpenCV window
        sys.exit(0) 
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}", file=sys.stderr)
        detected_text = "Analysis Error"

    global_dominant_emotion = detected_text
    full_text_to_display = f"Result: {global_dominant_emotion}"
    
    # --- Update Label Text and Adjust Size --- 
    emotion_label.setText(full_text_to_display)
    
    # Define constraints and padding
    horizontal_padding = 40
    vertical_padding = 20
    max_width_ratio = 0.8 # Max 80% of screen width
    max_width = int(screen_width * max_width_ratio)
    bottom_margin = 20 # Distance from bottom edge

    # Calculate ideal text width based on font
    font_metrics = QFontMetrics(emotion_label.font())
    # Use horizontalAdvance for single-line potential width
    ideal_text_width = font_metrics.horizontalAdvance(full_text_to_display)
    
    # Determine the actual width to use, respecting max_width
    new_width = min(ideal_text_width + horizontal_padding, max_width)

    # Calculate the required height *for the text when wrapped* within the new_width
    # Subtract padding first for accurate bounding rect calculation
    text_rect = font_metrics.boundingRect(0, 0, new_width - horizontal_padding, 0, 
                                          int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), 
                                          full_text_to_display)
    new_height = text_rect.height() + vertical_padding

    # Recalculate X and Y position based on new dimensions
    new_x = (screen_width - new_width) // 2
    new_y = screen_height - new_height - bottom_margin 
    
    # Apply new geometry to the overlay widget and the label inside
    overlay_widget.setGeometry(new_x, new_y, new_width, new_height)
    # Label needs internal padding, use margins or adjust geometry slightly if text hits edges
    emotion_label.setGeometry(horizontal_padding // 2, vertical_padding // 2, 
                              new_width - horizontal_padding, new_height - vertical_padding)
    # --- End Update Label Text and Adjust Size --- 
    
    cv2.waitKey(1) 

# --- Timer to run analysis periodically ---
# Analyze every 2000ms (2 seconds) for POC to reduce API calls
timer = QTimer()
timer.timeout.connect(analyze_screen)
timer.start(2000) # <-- Changed interval to 2 seconds

print("Starting screen analysis for emotion overlay. Press Ctrl+C in terminal to quit.")

# Start the PyQt event loop
exit_code = app.exec()
cv2.destroyAllWindows() # Ensure windows are closed on normal exit too
sys.exit(exit_code)
