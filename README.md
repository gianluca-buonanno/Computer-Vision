Real-Time Multi-Face Emotion and Age Analysis System
===================
Overview
This project is a real-time computer vision system that performs live facial detection, emotion recognition, age estimation, and temporal emotion trend analysis from a webcam feed. It is designed with performance, scalability, and robustness in mind, using multithreading, face tracking, and lightweight preprocessing to maintain smooth real-time operation.
The system can simultaneously track multiple faces, assign persistent IDs, analyze emotional states over time, and log structured emotion data for downstream analysis.

Key Features
===================
Real-Time Face Detection and Tracking
Uses dlib’s frontal face detector for fast and reliable face detection.
Assigns persistent IDs to faces using spatial proximity matching.
Automatically removes faces that disappear from view.

Emotion Recognition
===================
Integrates DeepFace for emotion inference.
Detects the dominant facial emotion per face.
Captures full emotion confidence distributions.
Supports multiple concurrent faces.

Temporal Emotion Trends
===================
Maintains a rolling emotion history per face.
Computes emotion trends based on recent observations.
Reduces noise from frame-by-frame emotion variation.

Emotion Intensity Classification
===================
Converts emotion confidence scores into qualitative levels: High, Medium, and Low.
Improves interpretability of real-time results and logs.

Age Estimation
===================
Estimates age per detected face using DeepFace.
Updates dynamically as faces remain in view.

High-Performance Architecture
===================
Uses a multithreaded design to separate video capture and rendering from heavy model inference.
Employs bounded queues to prevent frame backlog and latency spikes.
Emotion analysis is performed at fixed intervals to balance accuracy and performance.

Structured Logging
===================
Logs timestamped emotion events to a local log file.
Captured data includes face ID, dominant emotion, emotion intensity, estimated age, and frame number.

System Architecture Overview
===================
Webcam feed is captured in real time.
Frames are processed for face detection and tracking.
Selected frames are sent to a background analysis thread.
Emotion and age predictions are matched back to tracked faces.
Results are visualized live and logged for later analysis.

Preprocessing Pipeline
===================
Each frame undergoes lightweight preprocessing to improve robustness.
Frames are converted to grayscale, histogram equalization is applied, and the image is converted back to color format for compatibility with the analysis model.

Visual Output
===================
For each detected face, the system displays a bounding box with a unique color, face ID, dominant emotion, emotion intensity, estimated age, and emotion trend once enough samples have been collected.



Installation
===================
The system requires Python 3.8 or later and a connected webcam.
Required dependencies include DeepFace, OpenCV, dlib, and NumPy.
All dependencies can be installed via pip.

Usage
===================
Run the script from the command line to start live analysis.
The webcam feed will open automatically.
Press the q key to exit the application.

Example Log Output
===================
Timestamped entries record the frame number, face ID, detected emotion, intensity level, and estimated age.

Design Considerations
===================
Multithreading ensures that model inference does not block real-time video rendering.
Frame sampling balances responsiveness and accuracy.
Spatial matching avoids the overhead of more complex re-identification models.
Emotion trend smoothing improves stability of predictions.

Potential Extensions
===================
Gender estimation.
Persistent identity tracking using facial embeddings.
Database-backed analytics and dashboards.
API or web-based streaming interface.
Integration with behavioral research or user experience analysis tools.

Skills Demonstrated
===================
Computer vision.
Real-time system design.
Multithreaded Python programming.
AI model integration.
Performance optimization and logging.

License
===================
This project is intended for educational and portfolio use. Review third-party library licenses before any commercial deployment.
