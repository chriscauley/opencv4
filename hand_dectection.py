import cv2
import numpy as np
import math

HIST_FILE = 'hist.npy'

def capture_histogram(cap):
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1000, 600))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Place region of the hand inside box & press `A`',
                    (5, 50), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (500, 100), (580, 180), (105, 105, 105), 2)
        box = frame[105:175, 505:575]

        cv2.imshow("Capture Histogram", frame)
        key = cv2.waitKey(10)
        if key == ord('a'):
            object_color = box
            break
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    object_color_hsv = cv2.cvtColor(object_color, cv2.COLOR_BGR2HSV)
    object_hist = cv2.calcHist([object_color_hsv], [0, 1], None,
                               [12, 15], [0, 180, 0, 256])

    cv2.normalize(object_hist, object_hist, 0, 255, cv2.NORM_MINMAX)
    np.save(HIST_FILE, object_hist)
    return object_hist

def locate_object(frame, object_hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # apply back projection to image using object_hist as
    # the model histogram
    object_segment = cv2.calcBackProject(
      [hsv_frame], [0, 1], object_hist, [0, 180, 0, 256], 1)

    _, segment_thresh = cv2.threshold(object_segment, 70, 255, cv2.THRESH_BINARY)

    # apply some image operations to enhance image
    kernel = None
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    filtered = cv2.filter2D(segment_thresh, -1, disc)

    eroded = cv2.erode(filtered, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # masking
    masked = cv2.bitwise_and(frame, frame, mask=closing)

    return closing, masked, segment_thresh

def detect_hand(frame, hist):
    return_value = {}

    detected_hand, masked, raw = locate_object(frame, hist)
    return_value["binary"] = detected_hand
    return_value["masked"] = masked
    return_value["raw"] = raw

    # # find the contours
    image, contours, _ = cv2.findContours(detected_hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    palm_area = 0
    flag = None
    cnt = None

    # find the largest contour
    for (i, c) in enumerate(contours):
        area = cv2.contourArea(c)
        if area > palm_area:
            palm_area = area
            flag = i

    # we want our contour to have a minimum area of 10000
    # this number might be different depending on camera, distance of hand
    # from screen, etc.
    success = flag is not None and palm_area > 10000
    if success:
        cnt = contours[flag]
        return_value["contours"] = cnt
        cpy = frame.copy()
        cv2.drawContours(cpy, [cnt], 0, (0, 255, 0), 2)
        return_value["boundaries"] = cpy
    return success, return_value


def extract_fingertips(hand):
    cnt = hand["contours"]
    points = []
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    # get all the "end points" using the defects and contours
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        end = tuple(cnt[e][0])
        points.append(end)

    # filter out the points which are too close to each other
    filtered = filter_points(points, 50)

    # sort the fingertips in order of increasing value of the y coordinate
    filtered.sort(key=lambda point: point[1])

    # return the fingertips, at most 5.
    return [pt for idx, pt in zip(range(5), filtered)]


def filter_points(points, filterValue):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i] and points[j] and dist(points[i], points[j]) < filterValue:
                points[j] = None
    filtered = []
    for point in points:
        if point is not None:
            filtered.append(point)
    return filtered


def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (b[1] - a[1])**2)


def plot(frame, points):
    radius = 5
    colour = (0, 0, 255)
    thickness = -1
    for point in points:
        cv2.circle(frame, point, radius, colour, thickness)


if __name__ == "__main__":
    screen = np.zeros((600, 1000))
    curr = None
    prev = None
    cap = cv2.VideoCapture(0)
    try:
        hist = np.load(HIST_FILE)
    except FileNotFoundError:
        hist = capture_histogram(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        hand_detected, hand = detect_hand(frame, hist)

        if hand_detected:
            hand_image = hand["boundaries"]

            fingertips = extract_fingertips(hand)
            plot(hand_image, fingertips)

            prev = curr
            curr = fingertips[0]

            if prev and curr:
                cv2.line(screen, prev, curr, (255, 255, 255), 5)

            cv2.imshow("Drawing", screen)
            cv2.imshow("HD", hand_image)

        else:
            cv2.imshow("HD", frame)
        # for key, value in hand.items():
        #     if key == 'contours':
        #         continue
        #     cv2.imshow(key, value)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        if key == ord('c'):
            screen = np.zeros((600, 1000))
        if key == ord('h'):
            cv2.destroyAllWindows()
            hist = capture_histogram(cap)

    cap.release()
    cv2.destroyAllWindows()