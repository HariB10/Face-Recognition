from flask import Flask, render_template, request, redirect, url_for, Response, flash
import cv2
import os
import face_recognition
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

dataset_path = 'face_dataset'
encodings_file = 'face_encodings.pkl'

# Load the face encodings and names if they exist
if os.path.exists(encodings_file):
    with open(encodings_file, 'rb') as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]
else:
    known_encodings = []
    known_names = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    person_name = request.form['name']
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        os.makedirs(person_path)

    cap = cv2.VideoCapture(0)  # Initialize the webcam only for capturing images
    img_id = 0
    while img_id < 50:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            img_id += 1
            face = frame[y:y+h, x:x+w]  # Use the original frame (colored) instead of gray
            file_name = os.path.join(person_path, f"{img_id}.jpg")
            cv2.imwrite(file_name, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Draw name inside the rectangle
            cv2.putText(frame, person_name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Face Capture', frame)
        cv2.waitKey(1)

    cap.release()  # Release the webcam after capturing images
    cv2.destroyAllWindows()
    flash('Captured', 'capture')
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train():
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            # Load the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image {image_path}.")
                continue

            # Print the image shape for debugging
            print(f"Processing image: {image_path}, Shape: {image.shape}")

            try:
                # Convert the image to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get the face encodings
                encodings = face_recognition.face_encodings(rgb_image)

                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

    # Save the encodings and names
    data = {"encodings": known_encodings, "names": known_names}
    with open(encodings_file, 'wb') as f:
        pickle.dump(data, f)

    flash('Training Completed', 'train')
    return redirect(url_for('index'))



def gen_frames():
    cap = cv2.VideoCapture(0)  # Initialize the webcam only for real-time recognition
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Compare face encoding with known encodings
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            # Find best match
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

            # Draw name inside the rectangle
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()  # Release the webcam after real-time recognition
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/real_time_recognition')
def real_time_recognition():
    return render_template('real_time_recognition.html')

if __name__ == "__main__":
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    app.run(debug=True)
