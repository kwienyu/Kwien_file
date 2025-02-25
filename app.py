from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# OpenCV Video Capture
camera = cv2.VideoCapture(0)

# Global variable to track if the bounding box should be drawn
show_bounding_box = True


def extract_skin_tone(frame, bbox):
    """Extracts the average skin tone from the detected face region."""
    x, y, w, h = bbox

    # Extract face region
    face_region = frame[y:y + h, x:x + w]

    if face_region.size == 0:
        return (255, 255, 255)  # Default to white if extraction fails

    # Convert to RGB (OpenCV uses BGR)
    face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

    # Get average skin tone
    avg_color_per_row = np.mean(face_region, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)

    # Convert to integer BGR format
    skin_tone = tuple(map(int, avg_color[::-1]))  # Convert RGB to BGR

    return skin_tone


def generate_frames():
    """Captures camera feed and detects faces in real-time."""
    global show_bounding_box

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Extract and display skin tone
                skin_tone = extract_skin_tone(frame, (x, y, w, h))

                # Draw bounding box if enabled
                if show_bounding_box:
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Skin Tone: {skin_tone}", (x, y - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode frame to byte stream
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Renders the HTML page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Streams the camera feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_skin_tone')
def get_skin_tone():
    """Returns detected skin tone as JSON."""
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Camera capture failed'})

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            skin_tone = extract_skin_tone(frame, (x, y, w, h))
            return jsonify({'skin_tone': skin_tone})

    return jsonify({'error': 'No face detected'})


@app.route('/toggle_box', methods=['POST'])
def toggle_box():
    """Enables or disables the bounding box based on user request."""
    global show_bounding_box
    show_bounding_box = not show_bounding_box
    return jsonify({'status': 'success', 'show_bounding_box': show_bounding_box})


if __name__ == '__main__':
    app.run(debug=True)
