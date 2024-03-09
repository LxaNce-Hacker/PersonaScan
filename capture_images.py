# created by : LxaNce

import cv2
import dlib
import sys
import os

def capture_images(name):
    cap = cv2.VideoCapture(0)
    face_detector = dlib.get_frontal_face_detector()
    img_count = 0
    folder_path = f"demo_id/{name}/"

    print("Folder path:", folder_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_region = frame[y:y+h, x:x+w]
            img_count += 1
            img_filename = f"{folder_path}{name}_{img_count}.jpg"
            cv2.imwrite(img_filename, face_region)
            print(f"Saved image: {img_filename}")

        cv2.imshow("Capture Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python capture_images.py <person_name>")
    else:
        person_name = sys.argv[1]
        capture_images(person_name)
