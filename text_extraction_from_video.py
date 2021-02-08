# importing required libraries

from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract
import math
import os

# using pre-trained EAST model

net = cv2.dnn.readNet('frozen_east_text_detection.pb')


# function to display detected text

def text_detector(image, frame_number):

        # keeping a copy of originalinal image

    original = image
    (H, W) = image.shape[:2]

        # EAST requires input with dimensions which are multiple of 32

    (newW, newH) = (320, 320)

    # storing scaling factor for bounding box in original image

    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

        # sigmoid : confidence of detected text
        # concat_3 : geometry of detection

    layerNames = ['feature_fusion/Conv_7/Sigmoid',
                  'feature_fusion/concat_3']

        # normalizing with mean subtraction

    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (W, H),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
        )

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

        # getting dimensions

    for y in range(0, numRows):

        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns

        for x in range(0, numCols):

            # if our score does not have sufficient probability, ignore it

            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box

            endX = int(offsetX + cos * xData1[x] + sin * xData2[x])
            endY = int(offsetY - sin * xData1[x] + cos * xData2[x])
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # removing multile detections by choosing the best result
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # drawing bounding boxes
    for (startX, startY, endX, endY) in boxes:

        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        boundary = 2

        text = original[startY - boundary:endY + boundary, startX
                        - boundary:endX + boundary]
        try:
                text = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        except:
                continue
        textRecognized = '' + pytesseract.image_to_string(text)
        print ('Frame Number : ', frame_number)
        print(textRecognized)
        cv2.rectangle(original, (startX, startY), (endX, endY), (0,
                      255, 0), 3)
    return original


frame_number = 1

# reading test video
cap = cv2.VideoCapture('test_video.mp4')

# fout=open("output_file.txt","a")

frameRate = cap.get(11)

# Frame by Frame loop
while cap.isOpened():
    frameid = cap.get(1)
    (ret, frame) = cap.read()
    if ret:
        if frameid % 100 == 0:
            image = cv2.resize(frame, (640, 320),
                               interpolation=cv2.INTER_AREA)
            original = cv2.resize(frame, (640, 320),
                                  interpolation=cv2.INTER_AREA)
            #calling text detector function
            textDetected = text_detector(image, frame_number)
            #displaying image with and without bounding box
            cv2.imshow('originalinal Image', original)
            cv2.imshow('Text Detection', textDetected)
            time.sleep(2)
            frame_number += 1
    k = cv2.waitKey(30)
    if k == 27:
        break
cv2.destroyAllWindows()
