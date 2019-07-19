from imutils import paths
import os
import cv2
import numpy as np
import random

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, 
    help="file path to the images to be randomized")
args = vars(ap.parse_args())

for imagePath in paths.list_images(args["images"]):
    i = 0
    while i < 9 :
        image = cv2.imread(imagePath)

        rows, cols, yeet = image.shape

        flip = random.randrange(-2, 1)
        if flip != -2:
            image = cv2.flip(image, flip)

        rotate = random.randrange(0, 359)

        M = cv2.getRotationMatrix2D((cols/2, rows/2), rotate, 1)
        image = cv2.warpAffine(image, M, (cols, rows))

        imgPath = imagePath
        name = imgPath.split(os.path.sep).pop(-1)
        name = name[:name.find('.')]

        cv2.imwrite(name + " - " + str(i) + ".jpg", image)

        i += 1