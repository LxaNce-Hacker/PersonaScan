import dlib
import numpy as np
import os
import pickle

def train_face_recognition():
    dataset_path = "demo_id"
    output_embeddings_file = "embeddings.pkl"

    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    embeddings_dict = {}

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            person_embeddings = []

            for img_filename in os.listdir(person_path):
                img_path = os.path.join(person_path, img_filename)
                img = dlib.load_rgb_image(img_path)

                # Detect face in the image
                faces = face_detector(img)
                if len(faces) != 1:
                    print(f"Skipping {img_filename} due to no/multiple faces.")
                    continue

                face = faces[0]
                shape = shape_predictor(img, face)

                # Crop and resize the face image to match the expected size
                face_chip = dlib.get_face_chip(img, shape, size=150)
                
                # Compute face descriptor
                face_descriptor = face_recognizer.compute_face_descriptor(face_chip)
                person_embeddings.append(np.array(face_descriptor))

            if person_embeddings:
                embeddings_dict[person_name] = person_embeddings

    # Save the computed embeddings for later use
    with open(output_embeddings_file, "wb") as f:
        pickle.dump(embeddings_dict, f)

if __name__ == "__main__":
    train_face_recognition()
