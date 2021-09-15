import os
import time

import cv2 as cv
import face_recognition as fr

SCALE = 4
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def encode_faces(folder):
    encoded_faces = list()
    for f in os.listdir(folder):
        img_face = fr.load_image_file(f'{folder}/{f}')
        img_face = cv.resize(img_face, (0, 0), fx=1 / SCALE, fy=1 / SCALE)
        encodes = fr.face_encodings(img_face, num_jitters=5)
        print(f, len(encodes))
        encoded_face = encodes[0]  # there should be only one face in each face.
        encoded_faces.append((encoded_face, f))
    return encoded_faces


def mark_faces(img, coordinates_and_colored_labels):
    for coordinates_and_colored_label in coordinates_and_colored_labels:
        locations, label, color = coordinates_and_colored_label
        x1, y1, x2, y2 = locations
        x1 *= SCALE
        x2 *= SCALE
        y1 *= SCALE
        y2 *= SCALE
        cv.rectangle(img, (x2, y1), (x1, y2), color, 2)
        cv.rectangle(img, (x2 - 1, y2), (x1 + 1, y2 + 20), color, -1)
        cv.putText(img, label, (x2, y2 + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)


def init_recognition(folder, encoded_faces=[]):
    encodes = [x[0] for x in encoded_faces]
    labels = [x[1] for x in encoded_faces]
    if folder:
        init_recognition_from_folder(folder, encodes, labels)
    else:
        init_recognition_from_camera(encodes, labels)


def init_recognition_from_folder(folder, encodes, labels):
    for f in os.listdir(folder):
        img_face = fr.load_image_file(f'{folder}/{f}')
        smallimg = cv.resize(img_face, (0, 0), fx=1 / SCALE, fy=1 / SCALE)
        locations = fr.face_locations(smallimg)
        mark_faces(img_face, find_faces(smallimg, locations, encodes, labels))
        cv.imshow('Found Faces in ' + f, img_face)


def init_recognition_from_camera(encodes, labels):
    capture = cv.VideoCapture(0)
    p_time = 0
    stop_detecting = False
    last_detected_faces = list()
    while True:
        success, img = capture.read()
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        if success:
            smallimg = cv.resize(img, (0, 0), fx=1 / SCALE, fy=1 / SCALE)
            locations = fr.face_locations(smallimg)
            if stop_detecting and len(locations) != len(last_detected_faces):
                stop_detecting = False
            stop_detecting, detected_faces = find_faces(smallimg, locations, encodes, labels,
                                                        stop_detecting=stop_detecting, last_detected_faces=last_detected_faces)
            if stop_detecting:
                last_detected_faces = detected_faces
            mark_faces(img, detected_faces)
            cv.putText(img, f'FPS: {int(fps)}', (20, 80), cv.FONT_HERSHEY_PLAIN, 3, GREEN, 2)
            cv.imshow('Found Faces', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()


def find_faces(img, locations, known_face_encodings_only, known_face_labels_only, last_detected_faces=[],
               recognize=True, stop_detecting=False):
    coordinates_and_colored_labels = list()
    if recognize:
        if not stop_detecting:
            new_encodings = fr.face_encodings(img, num_jitters=5)
            # we found new faces and iterate over encodings to match the database.
            for new_encoding, location in zip(new_encodings, locations):
                label = "Unknown"
                compared_result = fr.compare_faces(known_face_encodings_only, new_encoding, tolerance=0.5)
                if True in compared_result:  # check if we can recognize this face
                    try:
                        label = known_face_labels_only[compared_result.index(True)]
                    except:
                        pass
                    color = RED
                else:
                    color = BLUE
                y1, x1, y2, x2 = location
                coordinates_and_colored_labels.append(((x1, y1, x2, y2), label, color))
            stop_detecting = True
        else:
            # detecting stopped.
            for last_detected_face, new_location in zip(last_detected_faces, locations):
                y1, x1, y2, x2 = new_location
                coordinates_and_colored_labels.append(((x1, y1, x2, y2), last_detected_face[1], last_detected_face[2]))
    else:
        for location in locations:  # we found new faces and iterate over encodings to match the database.
            label = "Unknown"
            color = (0, 120, 150)
            y1, x1, y2, x2 = location
            coordinates_and_colored_labels.append(((x1, y1, x2, y2), label, color))
    return stop_detecting, coordinates_and_colored_labels


if __name__ == '__main__':
    folder = "people"
    known_encoded_faces = encode_faces(folder)
    init_recognition('', known_encoded_faces)
