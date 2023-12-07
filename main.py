import cv2
import mediapipe as mp


def detect_hand_landmarks(img, hands):
    output_img = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image=output_img,
                                                      landmark_list=hand_landmarks,
                                                      connections=mp.solutions.hands.HAND_CONNECTIONS,
                                                      landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                                          color=(255, 255, 255), thickness=3, circle_radius=3),
                                                      connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                                          color=(0, 255, 0), thickness=3, circle_radius=3))
    return output_img, result


def count_fingers(img, result_hl):

    height, width, _ = img.shape
    output_img = img.copy()
    count = {'RIGHT': 0, 'LEFT': 0}

    finger_tip_id = [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                     mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                     mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                     mp.solutions.hands.HandLandmark.PINKY_TIP
                     ]

    finger_appeared = {'LEFT_THUMB': False,
                       'LEFT_INDEX': False,
                       'LEFT_MIDDLE': False,
                       'LEFT_RING': False,
                       'LEFT_PINKY': False,
                       'RIGHT_THUMB': False,
                       'RIGHT_INDEX': False,
                       'RIGHT_MIDDLE': False,
                       'RIGHT_RING': False,
                       'RIGHT_PINKY': False,
                       }

    for index, hand in enumerate(result_hl.multi_handedness):
        hand_label = hand.classification[0].label
        hand_landmarks = result_hl.multi_hand_landmarks[index]
        for tip_id in finger_tip_id:
            finger_name = tip_id.name.split("_")[0]
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                finger_appeared[hand_label.upper() + "_" + finger_name] = True
                count[hand_label.upper()] += 1
        thumb_tip_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP - 2].x
        if ((hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or
                (hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x))):
            finger_appeared[hand_label.upper() + "_THUMB"] = True
            count[hand_label.upper()] += 1

    cv2.putText(output_img, "Number of Fingers", (width // 2 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (184, 47, 87), 3)
    cv2.putText(output_img, str(sum(count.values())), (width // 2 - 130, 180), cv2.FONT_HERSHEY_SIMPLEX,
                6, (0, 255, 0), 12, 8)
    return output_img, finger_appeared, count


def paint_fingers(img, result_hl, fingers_st, count):
    output_img = img.copy()
    img_paths = {'left': ['left_notd.png'], 'right': ['right_notd.png']}
    if result_hl.multi_hand_landmarks:
        for index, hand in enumerate(result_hl.multi_handedness):
            hand_label = hand.classification[0].label
            img_paths[hand_label.lower()] = [hand_label.lower() + '_d.png']
            if count[hand_label.upper()] == 5:
                img_paths[hand_label.lower()] = [hand_label.lower() + '_allf.png']
            else:
                for finger, status in fingers_st.items():
                    if status and finger.split("_")[0] == hand_label.upper():
                        img_paths[hand_label.lower()].append(finger.lower() + '.png')
    return output_img, img_paths


def roi_draw(img, img_paths):
    for index, paths in enumerate(img_paths.values()):
        for img_path in paths:
            img_bgra = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            alpha = img_bgra[:, :, -1]
            img_bgr = img_bgra[:, :, :-1]
            height, width, _ = img_bgr.shape
            roi = img[15: 15 + height, (index * 870) + 15: ((index * 870) + 15 + width)]
            roi[alpha == 255] = img_bgr[alpha == 255]
            img[15: 15 + height, (index * 870) + 15: ((index * 870) + 15 + width)] = roi
    return img


landmarker = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)
cv2.namedWindow('Hand Detection & Finger Counting', cv2.WINDOW_NORMAL)

while camera_video.isOpened():
    fingers_statuses = {}
    counting = {}
    img_path = {}
    read, frame = camera_video.read()
    if not read:
        continue
    frame = cv2.flip(frame, 1)
    frame, results = detect_hand_landmarks(frame, landmarker)
    if results.multi_hand_landmarks:
        frame, fingers_statuses, counting = count_fingers(frame, results)
    frame, img_path = paint_fingers(frame, results, fingers_statuses, counting)
    frame = roi_draw(frame, img_path)
    cv2.imshow('Hand Detection & Finger Counting', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

camera_video.release()
cv2.destroyAllWindows()
