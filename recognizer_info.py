# recognizer_info.py
# created by : LxaNce

import cv2
import dlib
import numpy as np
import pickle
import json
import os
from datetime import datetime

# Load the trained face recognition model
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load the saved embeddings from the training phase
with open("embeddings.pkl", "rb") as f:
    embeddings_dict = pickle.load(f)

# Load additional information from JSON file
with open("additional_info.json", "r") as json_file:
    additional_info = json.load(json_file)

# Camera FullScreenCode
cap = cv2.VideoCapture(0)
screen_width = int(cap.get(3))  # Get the width of the camera feed
screen_height = int(cap.get(4))  # Get the height of the camera feed

## Full Screen 2lines
# cv2.namedWindow("Recognize Faces", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Recognize Faces", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define a global variable for storing attendance log
attendance_log = {}

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        shape = shape_predictor(gray, face)
        face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)
        detected_face_embedding = np.array(face_descriptor)

        recognized_person = None
        min_distance = float("inf")

        # Compare the detected face's embedding with the saved embeddings
        for person_id, person_embeddings in embeddings_dict.items():
            for person_embedding in person_embeddings:
                distance = np.linalg.norm(person_embedding - detected_face_embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_person = person_id

        if recognized_person is not None and min_distance < 0.6:
            # Look up additional information for the recognized person
            person_info = None
            for info in additional_info:
                if info["library_code"] == recognized_person:
                    person_info = info
                    break

            if person_info:
                # Display the recognized person's name and additional information on the frame
                info_lines = [
                    f"Id: {recognized_person}",
                    f"Name: {person_info['name']}",
                    f"Father's Name: {person_info['father_name']}",
                    f"Date Of Birth: {person_info['birth_date']}",
                    f"Address: {person_info['local_address']}",
                    f"Phone: {person_info['mobile_no']}",
                    f"Registration Date: {person_info['registration_date']}"
                ]
                line_height = 20
                y_position = face.top() - 10
                for line in info_lines:
                    cv2.putText(frame, line, (face.left(), y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    y_position += line_height

                # Update the attendance log
                if recognized_person not in attendance_log:
                    attendance_log[recognized_person] = {'name': person_info['name'], 'timestamp': str(datetime.now())}
                    print(f"Attendance marked for {person_info['name']} at {datetime.now()}")

            else:
                # Display only the recognized person's name if additional information is not available
                text = f"Recognized: {recognized_person}"
                cv2.putText(frame, text, (face.left(), face.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Recognize Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Print attendance log
print("Attendance Log:")
for person_id, info in attendance_log.items():
    print(f"{info['name']} - {info['timestamp']}")

cap.release()
cv2.destroyAllWindows()
