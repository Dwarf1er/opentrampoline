import cv2, mediapipe, time, numpy
mediapipe_drawing = mediapipe.solutions.drawing_utils
mediapipe_pose = mediapipe.solutions.pose

def bgr_to_rgb(current_frame):
    current_image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    current_image.flags.writeable = False
    return current_image

def rgb_to_bgr(current_image):
    current_image.flags.writeable = True
    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
    return current_image

def detect_pose(pose, current_image):
    results = pose.process(current_image)
    return results

def extract_landmarks(detected_pose):
    landmarks = detected_pose.pose_landmarks.landmark
    landmarks = {
        "nose" : landmarks[mediapipe_pose.PoseLandmark.NOSE.value],
        "left_eye_inner" : landmarks[mediapipe_pose.PoseLandmark.LEFT_EYE_INNER.value],
        "left_eye" : landmarks[mediapipe_pose.PoseLandmark.LEFT_EYE.value],
        "left_eye_outer" : landmarks[mediapipe_pose.PoseLandmark.LEFT_EYE.value],
        "right_eye_inner" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_EYE_INNER.value],
        "right_eye" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_EYE.value],
        "right_eye_outer" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_EYE_OUTER.value],
        "left_ear" : landmarks[mediapipe_pose.PoseLandmark.LEFT_EAR.value],
        "right_ear" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_EAR.value],
        "mouth_left" : landmarks[mediapipe_pose.PoseLandmark.MOUTH_LEFT.value],
        "mouth_right" : landmarks[mediapipe_pose.PoseLandmark.MOUTH_RIGHT.value],
        "left_shoulder" : landmarks[mediapipe_pose.PoseLandmark.LEFT_SHOULDER.value],
        "right_shoulder" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_SHOULDER.value],
        "left_elbow" : landmarks[mediapipe_pose.PoseLandmark.LEFT_ELBOW.value],
        "right_elbow" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_ELBOW.value],
        "left_wrist" : landmarks[mediapipe_pose.PoseLandmark.LEFT_WRIST.value],
        "right_wrist" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_WRIST.value],
        "left_pinky" : landmarks[mediapipe_pose.PoseLandmark.LEFT_PINKY.value],
        "right_pinky" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_PINKY.value],
        "left_index" : landmarks[mediapipe_pose.PoseLandmark.LEFT_INDEX.value],
        "right_index" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_INDEX.value],
        "left_thumb" : landmarks[mediapipe_pose.PoseLandmark.LEFT_THUMB.value],
        "right_thumb" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_THUMB.value],
        "left_hip" : landmarks[mediapipe_pose.PoseLandmark.LEFT_HIP.value],
        "right_hip" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_HIP.value],
        "left_knee" : landmarks[mediapipe_pose.PoseLandmark.LEFT_KNEE.value],
        "right_knee" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_KNEE.value],
        "left_ankle" : landmarks[mediapipe_pose.PoseLandmark.LEFT_ANKLE.value],
        "right_ankle" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_ANKLE.value],
        "left_heel" : landmarks[mediapipe_pose.PoseLandmark.LEFT_HEEL.value],
        "right_heel" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_HEEL.value],
        "left_foot_index" : landmarks[mediapipe_pose.PoseLandmark.LEFT_FOOT_INDEX.value],
        "right_foot_index" : landmarks[mediapipe_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
    }
    return landmarks

def draw_landmarks(current_image, detected_pose):
    mediapipe_drawing.draw_landmarks(
        current_image,
        detected_pose.pose_landmarks,
        mediapipe_pose.POSE_CONNECTIONS,
        mediapipe_drawing.DrawingSpec(color=(249,250,155), thickness=3, circle_radius=2),
        mediapipe_drawing.DrawingSpec(color=(255,119,119), thickness=3, circle_radius=2)
    )

def main():
    # jump counter variables
    counter = 0
    start_time = None
    end_time = None
    is_not_initialized = True
    is_not_time_started = True
    previous_left_shoulder = None
    previous_right_shoulder = None
    previous_left_hip = None
    previous_right_hip = None
    stage = None

    video_capture = cv2.VideoCapture(0)
    escape_key = 27
    with mediapipe_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while video_capture.isOpened():
            return_variable, current_frame = video_capture.read()

            # recolor to RGB
            current_image = bgr_to_rgb(current_frame)
            #detect pose
            detected_pose = detect_pose(pose, current_image)          
            # recolor to BGR
            current_image = rgb_to_bgr(current_image)

            # extract landmarks
            try:
                landmarks = extract_landmarks(detected_pose)

                # get coordinates
                current_left_shoulder = landmarks["left_shoulder"].y
                current_right_shoulder = landmarks["right_shoulder"].y
                current_left_hip = landmarks["left_hip"].y
                current_right_hip = landmarks["right_hip"].y

                # jump logic goes here
                if is_not_initialized:
                    previous_left_shoulder = current_left_shoulder
                    previous_right_shoulder = current_right_shoulder
                    previous_left_hip = current_left_hip
                    previous_right_hip = current_right_hip
                    is_not_initialized = False
                
                if current_left_shoulder < previous_left_shoulder and current_right_shoulder < previous_right_shoulder and current_left_hip < previous_left_hip and current_right_hip < previous_right_hip:
                    if is_not_time_started:
                        start_time = time.time()
                        is_not_time_started = False
                    stage = "up"
                
                if current_left_shoulder > previous_left_shoulder and current_right_shoulder > previous_right_shoulder and current_left_hip > previous_left_hip and current_right_hip > previous_right_hip and stage == "up":
                    stage = "down"
                    counter += 1

                if counter == 10 and end_time == None:
                    end_time = time.time()
                
                # render counter
                cv2.rectangle(current_image,
                    (0,0),
                    (455,73),
                    (245,117,16),
                    -1
                )

                cv2.putText(current_image,
                    "JUMPS",
                    (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA
                )

                cv2.putText(current_image,
                    str(counter),
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA
                )

                cv2.putText(current_image,
                    "STAGE",
                    (100,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA
                )

                cv2.putText(current_image,
                    stage,
                    (100,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA
                )

                cv2.putText(current_image,
                    "TIME",
                    (300,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA
                )

                if (end_time != None):
                    cv2.putText(current_image,
                        f"{end_time - start_time:.2f}",
                        (300,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA
                    )
                
                else:
                    cv2.putText(current_image,
                        "0",
                        (300,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA
                    ) 

            except:
                pass

            # render detections
            draw_landmarks(current_image, detected_pose)

            cv2.imshow("OpenTrampoline", current_image)

            if cv2.waitKey(10) == escape_key:
                break
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()