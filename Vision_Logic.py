import cv2

# Function to find faces in a frame
def find_faces_manually(current_frame, face_model):
  # Step 1: Make it grayscale (Beginners do this for better detection)
  gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

  # Step 2: Look for faces
  # scaleFactor = 1.1 means it looks for different sizes of faces
  all_faces = face_model.detectMultiScale(gray, 1.1, 5)

  return all_faces
