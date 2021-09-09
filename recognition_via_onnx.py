import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import cv2 as cv
import os
import time
import numpy as np
def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def encode_faces(folder):
    pass
    # encoded_faces = list()
    # for f in os.listdir(folder):
    #     img_face = fr.load_image_file(f'{folder}/{f}')
    #     encodes = fr.face_encodings(img_face, num_jitters=5)
    #     print(f, len(encodes))
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
    #     encoded_face = encodes[0]  # there should be only one face in each face.
    #     encoded_faces.append((encoded_face, f))
    # return encoded_faces


def init_recognition(folder, encoded_faces=[]):
    # if folder:
    #     for f in os.listdir(folder):
    #         img_face = fr.load_image_file(f'{folder}/{f}')
    #         find_faces(img_face, encoded_faces)
    #         cv.imshow('Found Faces in ' + f, img_face)
    #     cv.waitKey()
    # else:
    onnx_model = onnx.load('ultra_light/ultra_light_models/Mb_Tiny_RFB_FD_train_input_640.onnx')
    predictor = prepare(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    capture = cv.VideoCapture(0)
    p_time = 0
    while True:
        success, frame = capture.read()
        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time
        if success:
            # find_faces(img, encoded_faces)
            # lets find height and width
            h, w, _ = frame.shape
            # BGR to RGB( default color is BGR)
            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # let's resize the img(640, 480)
            img = cv.resize(img, (640, 480))
            img_mean = np.array([127, 127, 127])
            img = (img - img_mean) / 128
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)
            confidences, boxes = ort_session.run(None, {input_name: img})
            boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
            boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                x1, y1, x2, y2 = box
                cv.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
                cv.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv.FILLED)
                font = cv.FONT_HERSHEY_DUPLEX
                text = f"face: {labels[i]}"
                cv.putText(frame, text, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)

            cv.putText(frame, f'FPS: {int(fps)}', (20,80), cv.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
            cv.imshow('Found Faces', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()

#
# def find_faces(img, known_face_encodings_and_labels=[]):
#     new_encodings = fr.face_encodings(img, num_jitters=5)
#     known_face_encodings_only = [x[0] for x in known_face_encodings_and_labels]
#     known_face_labels_only = [x[1] for x in known_face_encodings_and_labels]
#     face_index = 0
#     locations = fr.face_locations(img)
#     for new_encoding in new_encodings:  # we found new faces and iterate over encodings to match the database.
#         label = "Unknown"
#         if face_index < len(locations):
#             compared_result = fr.compare_faces(known_face_encodings_only, new_encoding, tolerance=0.5)
#             if True in compared_result:  # check if we can recognize this face
#                 try:
#                     label = known_face_labels_only[compared_result.index(True)]
#                 except:
#                     pass
#                 color = (255, 0, 0)
#             else:
#                 color = (0, 0, 255)
#             y1, x1, y2, x2 = locations[face_index]
#             cv.rectangle(img, (x2, y1), (x1, y2), color, 2)
#             cv.rectangle(img, (x2 - 1, y2), (x1 + 1, y2 + 20), color, -1)
#             cv.putText(img, label, (x2, y2 + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#         face_index += 1


if __name__ == '__main__':
    folder = "people"
    known_encoded_faces = encode_faces(folder)
    init_recognition('', known_encoded_faces)
