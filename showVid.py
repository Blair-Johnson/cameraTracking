import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help = "Path to test file", type = str)
    return parser.parse_args()

def process(img):
    img = cv2.resize(img, (640,480))
    return img

def capture_loop(path):

    capture = cv2.VideoCapture(path)

    while capture.isOpened():
        ret, img = capture.read()

        if ret == False:
            break

        #main loop of processing
        img = process(img)
        #displaying results
        cv2.imshow("Stream", img)
        cv2.waitKey(1)
        if cv2.getWindowProperty('Stream',cv2.WND_PROP_VISIBLE) < 1:        
            break

    capture.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    args = get_args()
    
    capture_loop(args.path)

    print('done!')
