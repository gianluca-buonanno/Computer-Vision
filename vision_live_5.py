from deepface import DeepFace
import cv2
import numpy as np
import dlib
import os
from collections import deque, Counter
import threading
import queue
import logging

# Preprocessing function for frames
def preprocess_frame(frame):
    """Apply lightweight preprocessing to enhance frame quality for analysis."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# Class to manage tracked faces
class TrackedFace:
    def __init__(self, id, bbox):
        """Initialize a tracked face with ID and bounding box."""
        self.id = id                    # Unique identifier for the face
        self.bbox = bbox                # Bounding box: (x, y, w, h)
        self.emotion_history = deque(maxlen=10)  # History of recent emotions
        self.dominant_emotion = None    # Latest detected emotion
        self.scores = {}                # Emotion confidence scores
        self.age = None                 # Estimated age
        self.last_seen = 0              # Frame count when last detected
        self.missed_frames = 0          # Frames missed since last detection
        self.color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color for visualization

# Helper function to determine intensity
def get_intensity(score):
    """Determine the intensity of the emotion based on the confidence score."""
    if score > 70:
        return "High"
    elif score > 40:
        return "Medium"
    else:
        return "Low"

# Thread function for emotion and age analysis
def analyze_emotion_thread(frame_queue, result_queue):
    """Perform emotion and age analysis in a separate thread to avoid blocking."""
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:  # Signal to exit thread
                break
            # Analyze emotions and age (no gender)
            results = DeepFace.analyze(frame, actions=['emotion', 'age'], 
                                       detector_backend='opencv', enforce_detection=False)
            analysis_results = []
            for result in results:
                region = result['region']              # Bounding box from DeepFace
                dominant = result['dominant_emotion']  # Most prominent emotion
                scores = result['emotion']             # Emotion scores dictionary
                age = result['age']                    # Estimated age
                analysis_results.append((region, dominant, scores, age))
            result_queue.put(analysis_results)
        except queue.Empty:
            continue  # Wait for frame if queue is empty
        except Exception as e:
            result_queue.put([{"error": str(e)}])  # Report errors

# Main function for live emotion detection
def analyze_emotion_live():
    """Run live emotion detection with multiple face tracking, trends, age, and intensity."""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Queues for communication between threads
    frame_queue = queue.Queue(maxsize=1)   # Limit to 1 to avoid backlog
    result_queue = queue.Queue(maxsize=1)

    # Start analysis thread
    analysis_thread = threading.Thread(
        target=analyze_emotion_thread,
        args=(frame_queue, result_queue),
        daemon=True
    )
    analysis_thread.start()

    # Variables for tracking
    tracked_faces = []        # List of tracked faces
    frame_count = 0           # Frame counter
    ANALYSIS_INTERVAL = 5     # Analyze every 5 frames

    # Set up logging
    logging.basicConfig(filename='emotion_log.log', level=logging.INFO, 
                        format='%(asctime)s - %(message)s')

    print("Starting live emotion analysis... (Press 'q' to quit)")

    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = preprocess_frame(frame)  # Preprocess for analysis

        # Detect faces using dlib
        faces = detector(gray)
        current_faces = []

        # Match detected faces to tracked faces
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            center = (x + w/2, y + h/2)
            matched_tf = None
            min_dist = float('inf')

            # Find closest tracked face within threshold
            for tf in tracked_faces:
                tf_center = (tf.bbox[0] + tf.bbox[2]/2, tf.bbox[1] + tf.bbox[3]/2)
                dist = np.linalg.norm(np.array(center) - np.array(tf_center))
                if dist < min_dist and dist < 50:  # Threshold of 50 pixels
                    min_dist = dist
                    matched_tf = tf

            if matched_tf:
                # Update existing tracked face
                matched_tf.bbox = (x, y, w, h)
                matched_tf.last_seen = frame_count
                matched_tf.missed_frames = 0
            else:
                # Add new tracked face
                new_id = len(tracked_faces) + 1
                tracked_faces.append(TrackedFace(new_id, (x, y, w, h)))

            current_faces.append((x, y, w, h))

        # Update missed frames and remove lost faces
        for tf in tracked_faces[:]:
            if tf.last_seen < frame_count:
                tf.missed_frames += 1
                if tf.missed_frames > 10:  # Remove after 10 missed frames
                    tracked_faces.remove(tf)

        # Periodically send frame for analysis
        if frame_count % ANALYSIS_INTERVAL == 0 and frame_queue.empty():
            frame_queue.put(processed_frame)

        # Process analysis results
        while not result_queue.empty():
            analysis_results = result_queue.get_nowait()
            if (isinstance(analysis_results, list) and 
                len(analysis_results) > 0 and 
                "error" in analysis_results[0]):
                print("Error in analysis:", analysis_results[0]["error"])
                continue

            for region, dominant, scores, age in analysis_results:
                r_center = (region['x'] + region['w']/2, region['y'] + region['h']/2)
                min_dist = float('inf')
                matched_tf = None

                # Match analysis result to tracked face
                for tf in tracked_faces:
                    tf_center = (tf.bbox[0] + tf.bbox[2]/2, tf.bbox[1] + tf.bbox[3]/2)
                    dist = np.linalg.norm(np.array(r_center) - np.array(tf_center))
                    if dist < min_dist and dist < 50:
                        min_dist = dist
                        matched_tf = tf

                if matched_tf:
                    matched_tf.dominant_emotion = dominant
                    matched_tf.scores = scores
                    matched_tf.emotion_history.append(dominant)
                    matched_tf.age = age

                    # Log the data
                    intensity = get_intensity(scores[dominant])
                    logging.info(f"Frame {frame_count}, Face ID {matched_tf.id}, "
                                 f"Emotion: {dominant}, Intensity: {intensity}, Age: {age}")

        # Display results for each tracked face
        for tf in tracked_faces:
            x, y, w, h = tf.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), tf.color, 2)
            if tf.dominant_emotion and tf.age is not None:
                intensity = get_intensity(tf.scores[tf.dominant_emotion])
                text = f"ID {tf.id}: {tf.dominant_emotion} ({intensity}), Age: {tf.age}"
                if len(tf.emotion_history) >= 5:  # Show trend after 5 samples
                    trend = Counter(tf.emotion_history).most_common(1)[0][0]
                    text += f" (Trend: {trend})"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, tf.color, 2)

        # Show frame
        cv2.imshow('Live Emotion Detection', frame)
        frame_count += 1

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    frame_queue.put(None)  # Signal thread to exit
    analysis_thread.join()
    cap.release()
    cv2.destroyAllWindows()

# Entry point with package check
def main():
    """Check for required packages and start the program."""
    try:
        import deepface
        import dlib
    except ImportError:
        print("Installing required packages...")
        os.system('pip install deepface opencv-python dlib')
        print("Please re-run the script after installation.")
        exit(1)

    analyze_emotion_live()

if __name__ == "__main__":
    main()