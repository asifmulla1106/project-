import os.path
from skimage.feature import hog
import imutils
import numpy as np
import cv2


def hog_detector(X):
    ppc = 16
    hog_images = []
    hog_features = []
    for image in X:
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), block_norm='L2',
                            visualize=True)
        hog_images.append(hog_image)
        hog_features.append(fd)

    hog_features = np.array(hog_features)
    return hog_features


def detect(frame):
    bounding_box, _ = HOGCV.detectMultiScale(frame, winStride=(4, 4), scale=1.03)
    human = 1
    for startx, starty, w, h in bounding_box:
        cv2.rectangle(frame, (startx, starty), (startx + w, starty + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Human {human}', (startx, startx), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        human += 1

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f'Total Human : {human - 1}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    cv2.imshow('Webcame', frame)
    return frame


def start_camera(camera_id=0):
    video = cv2.VideoCapture(camera_id)
    print('Detecting Human...')
    while True:
        check, frame = video.read()
        detect(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def read_video_file(path):
    video_file = cv2.VideoCapture(path)
    ret, frame = video_file.read()

    if not ret:
        print('Video Not Found. Please pass a right video file inorder to run.')
        return
    print('Detecting Human...')

    while video_file.isOpened():
        check, frame = video_file.read()

        if check:
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            detect(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video_file.release()
    cv2.destroyAllWindows()


def main(webcame=True, video=None):
    """

    :Asif webcame: If you want to use webcame. Simply pass webcame = true it will start the webcame and start reading the image
    :Asif video:  : If you want to test it with a video file pass the file path and it will read from it
    :return:
    """

    if webcame:
        start_camera()
    elif video is not None:
        if os.path.exists(video):
            read_video_file(video)
        else:
            print(f'Video File {video} Passed Cannot be found please pass correct video file')
    else:
        print('Set Webcame equal to True or pass a video file.')


if __name__ == '__main__':
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # For reading from file
    main(webcame=True, video='./video.mp4')

    # For reading from camera
    # main()
