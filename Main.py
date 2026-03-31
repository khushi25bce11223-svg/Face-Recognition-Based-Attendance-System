import cv2
import Setup_Config as config
import Data_Handler as data_tool
import Vision_Logic as vision_tool

# 1. Setup the tools
cap = cv2.VideoCapture(config.MY_CAMERA_NUMBER)
path = cv2.data.haarcascades + config.DETECTOR_DATA
face_cascade = cv2.CascadeClassifier(path)

print("VIT ATTENDANCE SYSTEM IS STARTING")
print("Press 'q' to stop the camera and exit.")

while True:
  # Capture a frame
  ret, frame = cap.read()
  if not ret:
    break

  # Use our vision part to find faces
  faces_found = vision_tool.find_faces_manually(frame, face_cascade)

  #Loop through each face found
  for i in range(len(faces_found)):
    (x, y, w, h) = faces_found[i]

    # Draw a rectangle around the face
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, "Recording...", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Use our data part to save the time
    data_tool.save_presence_to_csv(config.OUTPUT_FILE)

 # Show the video feed
 cv2.imshow('Attendance Feed', frame)

 # Quit logic
 if cv2.waitKey(1) & 0xFF == ord('q'):
   break


#Cleanup
cap.release()
cv2.destroyAllWindows()
print("System closed. Attendance is saved.")

