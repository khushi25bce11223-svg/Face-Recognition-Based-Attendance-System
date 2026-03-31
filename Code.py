import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import csv
import pickle

# SETTINGS
PATH_TO_IMAGES = 'student_images'
ENCODING_FILE = 'known_faces.pickle'
ATTENDANCE_LOG_DIR = 'attendance_logs'
ATTENDANCE_COOLDOWN = 600 
attendance_tracker = {} 

# To Create folders if they don't exist
if not os.path.exists(PATH_TO_IMAGES):
    os.makedirs(PATH_TO_IMAGES)
if not os.path.exists(ATTENDANCE_LOG_DIR):
    os.makedirs(ATTENDANCE_LOG_DIR)

# STEP 1: LOAD IMAGES AND CREATE ENCODINGS
def create_and_save_encodings():
    known_face_encodings = []
    known_face_names = []

    print("Step 1: Looking for images in the folder")
    
    file_list = os.listdir(PATH_TO_IMAGES)
    if not file_list:
        print("Error: No images found! Add some .jpg or .png files first.")
        return

    for filename in file_list:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Get the name from the filename
            name = os.path.splitext(filename)[0].replace("_", " ")
            
            # Load the image
            path = os.path.join(PATH_TO_IMAGES, filename)
            image = face_recognition.load_image_file(path)
            
            # Find the face encoding
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"Success: Encoded {name}")
            else:
                print(f"Warning: No face found in {filename}. Skipping.")

    # Save everything to a file so that we don't have to do this every time
    data = {"encodings": known_face_encodings, "names": known_face_names}
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(data, f)
    print("Step 1 Complete: All encodings saved to disk.\n")

# STEP 2: MARK ATTENDANCE IN CSV
def mark_attendance(name):
    current_time = datetime.now().strftime('%H:%M:%S')
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Check for different person
    if name in attendance_tracker:
        last_time = attendance_tracker[name]
        if (datetime.now().timestamp() - last_time) < ATTENDANCE_COOLDOWN:
            return # Skip marking

    # Log it
    log_path = os.path.join(ATTENDANCE_LOG_DIR, f'{current_date}_Attendance.csv')
    file_exists = os.path.isfile(log_path)

    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Time', 'Date', 'Status'])
        writer.writerow([name, current_time, current_date, 'PRESENT'])
    
    attendance_tracker[name] = datetime.now().timestamp()
    print(f"Done! Marked {name} present at {current_time}")

# STEP 3: MAIN LOOP (THE WEBCAM)
def start_attendance_system():
    # Try to load existing encodings
    try:
        with open(ENCODING_FILE, "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
    except FileNotFoundError:
        print("No encoding file found. Creating one now...")
        create_and_save_encodings()
        return start_attendance_system()

    cap = cv2.VideoCapture(0)
    print("Webcam starting... Press 'q' to stop.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Shrink image to make it run faster
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find faces in current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for encode_face, face_loc in zip(face_encodings, face_locations):
            # Compare face with our database
            matches = face_recognition.compare_faces(known_face_encodings, encode_face)
            face_distances = face_recognition.face_distance(known_face_encodings, encode_face)
            
            name = "Unknown"
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                # If the match is better than 0.6 distance (lower is better)
                if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]
                    mark_attendance(name)

            # Draw the box
            y1, x2, y2, x1 = [v * 4 for v in face_loc]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Finally Run the program
if __name__ == "__main__":
    start_attendance_system()
