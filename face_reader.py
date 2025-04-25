import cv2
import sys

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

# Start video capture from the default webcam (usually index 0)
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.", file=sys.stderr)
    sys.exit(1)

print("Starting webcam feed. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame.", file=sys.stderr)
        break # Exit if we can't capture a frame

    # Convert the frame to grayscale (Haar cascades work better on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # scaleFactor: How much the image size is reduced at each image scale.
    # minNeighbors: How many neighbors each candidate rectangle should have to retain it.
    # minSize: Minimum possible object size. Objects smaller than this are ignored.
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30) 
    )

    # Draw a rectangle around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green rectangle

    # Display the resulting frame
    cv2.imshow('Face Reader', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy windows
video_capture.release()
cv2.destroyAllWindows()
print("Webcam feed stopped.")
