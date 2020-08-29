import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help = "Path to test file", type = str)
    parser.add_argument("--max_points", help = "Maximum number of keypoints", default = 500, type = int)
    return parser.parse_args()

def get_corners(img, args):
    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    greyscale_img = cv2.convertScaleAbs(greyscale_img)
    corners = cv2.goodFeaturesToTrack(greyscale_img, maxCorners=args.max_points, qualityLevel=.01, minDistance=4)
    
    return corners
  
def convert_to_keypoints(corners):
    kpts = [cv2.KeyPoint(x,y,4) for [[x,y]] in corners]
    return kpts

def gen_rectangle(corner, size = 4):
    size = int(size**.5)
    corner2 = [[x+size,y+size] for x,y in corner]
    xmin = int(corner[0][0])
    ymin = int(corner[0][1])
    xmax = int(corner2[0][0])
    ymax = int(corner2[0][1])

    return  (xmin, ymin), (xmax, ymax)

def plot_corners(img, corners):
    for corner in corners:
        topl, btr = gen_rectangle(corner)
        img = cv2.rectangle(img, topl, btr, color=(255,0,0), thickness=1)

    return img

def extract_keypoints(img, corners):

    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    greyscale_img = cv2.convertScaleAbs(greyscale_img)
    
    orb = cv2.ORB_create()
    kpts = convert_to_keypoints(corners)
    kpts, des = orb.compute(greyscale_img, kpts)

    return kpts, des

def track(des0, des1):
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf_matcher.match(des0, des1)
    
    return matches

def show_matches(kpts0, kpts1, matches, img):
    for match in matches:
        src = kpts1[match.trainIdx].pt
        dst = kpts0[match.queryIdx].pt
        src = tuple([int(i) for i in src])
        dst = tuple([int(i) for i in dst])
        img = cv2.line(img, src, dst, color=(255,0,255), thickness=2)
    return img



def process(img1, args, des0 = None, kpts0 = None):
    img1 = cv2.resize(img1, (853,480))
    
    corners = get_corners(img1, args)
    img = plot_corners(img1, corners)

    kpts1, des1 = extract_keypoints(img1, corners)

    if type(des0) != type(None):
        matches = track(des0, des1)
        matches = [match for match in matches if match.distance < 10]
        print(len(matches))
        #matches = sorted(matches, key = lambda x:x.distance)
        #matches = matches[:int(len(matches)*.40)]
        img = show_matches(kpts0, kpts1, matches, img)
        # for each match I want to draw a line between the train descriptor (des0) location
        # and the match descriptor (des1) location. This should allow me to see the motion
        # of trackers as the scene progresses

    return img, des1, kpts1

def capture_loop(args):
    closed = False
    capture = cv2.VideoCapture(args.path)
    
    init = False
    while capture.isOpened():
        if not init:
            ret, img = capture.read()

            if not ret:
                break

            img, des0, kpts0 = process(img, args)

            cv2.imshow("Stream", img)
            cv2.waitKey(1)
            if cv2.getWindowProperty('Stream',cv2.WND_PROP_VISIBLE) < 1:
                closed = True
                break

            init = True

        ret, img = capture.read()

        if not ret:
            break

        img, des1, kpts1 = process(img, args, des0 = des0, kpts0 = kpts0)
        des0 = des1
        kpts0 = kpts1

        cv2.imshow("Stream", img)
        cv2.waitKey(1)
        if cv2.getWindowProperty('Stream',cv2.WND_PROP_VISIBLE) < 1:        
            closed = True
            break

    capture.release()
    cv2.destroyAllWindows()
    return closed

if __name__ == "__main__":
    args = get_args()
    
    terminate = False
    while not terminate:
        terminate = capture_loop(args)

    print('done!')
