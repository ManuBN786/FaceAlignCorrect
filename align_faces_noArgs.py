# USAGE : Download "shape_predictor_68_face_landmarks.dat" file from "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" and add image path. In the terminal do 'python3 align_faces_noArgs.py'
# Author: Manu BN

# import the necessary packages
import sys

sys.path.append('/usr/local/lib/python3.5/site-packages')
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2



# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread("/home/user/Alignment/images/2.jpg")
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original input image and detect faces in the grayscale
# image
cv2.imshow("Input", image)
rects = detector(gray, 2)

# loop over the face detections
for rect in rects:
	# extract the ROI of the *original* face, then align the face
	# using facial landmarks
	(x, y, w, h) = rect_to_bb(rect)
	faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
	faceAligned = fa.align(image, gray, rect)

	import uuid
	f = str(uuid.uuid4())
	cv2.imwrite("foo/" + f + ".png", faceAligned)

	# display the output images
	cv2.imshow("Original", faceOrig)
	cv2.imshow("Aligned", faceAligned)
        #x = cv2.imwrite('Aligned10.png',faceAligned)
      
	cv2.waitKey(0)
