import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
#from twilio.rest import Client

# Mask-RCNN config
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  
    DETECTION_MIN_CONFIDENCE = 0.6 #setted to 60%


# Filter to only cars
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)
'''
# Twilio config
twilio_account_sid = 'aaaaaaaaaaaaaaaaaaaaaaa'
twilio_auth_token = 'aaaaaaaaaaaaaaaaaaa93113efe'
twilio_phone_number = '+199999999'
destination_phone_number = '+19999999999'
client = Client(twilio_account_sid, twilio_auth_token)
'''

# Root dir
ROOT_DIR = Path(".")

#Trained model loc
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

VIDEO_SOURCE = "demo.mp4"

model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

model.load_weights(COCO_MODEL_PATH, by_name=True)

# spotted parking spaces
parked_car_boxes = None

video_capture1 = cv2.VideoCapture(VIDEO_SOURCE)

free_space_frames = 0
sms_sent = False
count = 0
temp = np.array(4,)

parked_car_boxes1 = [None] * 11

def checkEqual2(iterator):
   print(iterator)
#     return len(set(iterator)) <= 1
  
fval, fframe = video_capture1.read()
rgb_image = fframe[:, :, ::-1]
results = model.detect([rgb_image], verbose=0)
r = results[0]
print('got frame 1')

#parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])

video_capture = cv2.VideoCapture("test.mp4")

# Loop over each frame in the video
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        print("couldn't read video")
        break

    # Converting the image from BGR color used by OpenCV to RGB color
    #rgb_image = frame[:, :, ::-1]

    #results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    #r = results[0]

    # The r variable will now have the results of detection:
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)
    if parked_car_boxes is None:
        # This is the first frame of video - assume all the cars detected are in parking spaces.
        # Save the location of each car as a parking space box and go to the next frame of video.
        parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        print('got frame 1 spots')

    else:

        rgb_image = frame[:, :, ::-1]

        results = model.detect([rgb_image], verbose=0)

        # Mask R-CNN assumes we are running detection on multiple images.
        # We only passed in one image to detect, so only grab the first result.
        r = results[0]


        #video_capture = cv2.VideoCapture("test.mp4")
        #success, frame = video_capture.read()

        # Get where cars are currently located in the frame
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])

        # See how much those cars overlap with the known parking spaces
        overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)

        # Assume no spaces are free until we find one that is free
        free_space = False

        # Loop through each known parking space box
        for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):

            # For this parking space, find the max amount it was covered by any
            # car that was detected in our image (doesn't really matter which car)
            max_IoU_overlap = np.max(overlap_areas)

            # Get the top-left and bottom-right coordinates of the parking area
            y1, x1, y2, x2 = parking_area

            # Check if the parking space is occupied by seeing if any car overlaps
            # it by more than 0.15 using IoU
            if max_IoU_overlap < 0.15:
                # Parking space not occupied! Draw a green box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Flag that we have seen at least one open space
                free_space = True
            else:
                # Parking space is still occupied - draw a red box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Write the IoU measurement inside the box
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

        # If at least one space was free, start counting frames
        # This is so we don't alert based on one frame of a spot being open.
        # This helps prevent the script triggered on one bad detection.
        if free_space:
            free_space_frames += 1
        else:
            # If no spots are free, reset the count
            free_space_frames = 0

        # If a space has been free for several frames, we are pretty sure it is really free!
        if free_space_frames > 190:
            # Write SPACE AVAILABLE!! at the top of the screen
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"SPACE AVAILABLE!", (10, 150), font, 3.0, (0, 255, 0), 2, cv2.FILLED)
            
            # If we haven't sent an SMS yet, sent it!
            if not sms_sent:
                print("SENDING SMS!!!")
                print("Hope you got the message on your phone")

        # Show the frame of video on the screen
#         cv2.imshow('Video', frame)
    #saving each frame
    name = str(count) + ".jpg"
    name = os.path.join('./bk', name)
    cv2.imwrite(name, frame)
    count+=1
    
    #'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
print("Video finished")
video_capture.release()
# cv2.destroyAllWindows()

#create video including all the frames in the bk folder
import glob

images = list(glob.iglob(os.path.join('./bk', '*.*')))
images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

# Get all image file paths to a list.
# Sort the images by name index.
# images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

def make_video(outvid, images=None, fps=30, size=None,
               is_color=True, format="FMP4"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid
  
make_video('vidout2.mp4', images, fps=30)


