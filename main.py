import face_recognition as fr
import cv2 as cv
import os


def encode_faces(folder):
    encoded_faces = list()
    for f in os.listdir(folder):
        img_face = fr.load_image_file(f'{folder}/{f}')
        encodes = fr.face_encodings(img_face, num_jitters=5)
        print(f, len(encodes))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        encoded_face = encodes[0]  # there should be only one face in each face.
        encoded_faces.append((encoded_face, f))
    return encoded_faces


def init_recognition(folder, encoded_faces=[]):
    if folder:
        for f in os.listdir(folder):
            img_face = fr.load_image_file(f'{folder}/{f}')
            find_faces(img_face, encoded_faces)
            cv.imshow('Found Faces in ' + f, img_face)
        cv.waitKey()
    else:
        capture = cv.VideoCapture(0)
        while True:
            success, img = capture.read()
            if success:
                find_faces(img, encoded_faces)
                cv.imshow('Found Faces', img)
            cv.waitKey(5)


def find_faces(img, known_face_encodings_and_labels=[]):
    new_encodings = fr.face_encodings(img, num_jitters=5)
    known_face_encodings_only = [x[0] for x in known_face_encodings_and_labels]
    known_face_labels_only = [x[1] for x in known_face_encodings_and_labels]
    face_index = 0
    locations = fr.face_locations(img)
    for new_encoding in new_encodings:  # we found new faces and iterate over encodings to match the database.
        label = "Unknown"
        if face_index < len(locations):
            compared_result = fr.compare_faces(known_face_encodings_only, new_encoding, tolerance=0.5)
            if True in compared_result:  # check if we can recognize this face
                try:
                    label = known_face_labels_only[compared_result.index(True)]
                except:
                    pass
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            y1, x1, y2, x2 = locations[face_index]
            cv.rectangle(img, (x2, y1), (x1, y2), color, 2)
            cv.rectangle(img, (x2 - 1, y2), (x1 + 1, y2 + 20), color, -1)
            cv.putText(img, label, (x2, y2 + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        face_index += 1


if __name__ == '__main__':
    folder = "people"
    known_encoded_faces = encode_faces(folder)
    init_recognition('', known_encoded_faces)
