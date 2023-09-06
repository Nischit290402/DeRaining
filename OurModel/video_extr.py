# Get Frames
# Instr: Make directories as necessary

import cv2
import os

# video_filename = "E:\Minor_Project\Datasets\sampleRain.mp4"
# output_folder = "E:\Minor_Project\Datasets\sampleRain_Images"

def breakVideo(video_filename, output_folder):
    cap = cv2.VideoCapture(video_filename)
    
    # create a temporary folder to store the frames
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # loop through the video frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output_filename = os.path.join(output_folder, 'frame_{:06d}.jpg'.format(frame_count))
        cv2.imwrite(output_filename, frame)
        frame_count += 1

    cap.release()


