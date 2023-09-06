import cv2
import os

# input_folder = 'E:\Minor_Project\Datasets\sampleRain_Images'

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30
frame_size = (3840, 2160)

def makeVideo(input_folder, output_video):
    # Create the output video file
    output_video = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # Loop over the image files in the input folder and add them to the output video
    for image_filename in os.listdir(input_folder):
        if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
            image_path = os.path.join(input_folder, image_filename)
            frame = cv2.imread(image_path)
            if frame is not None:
                output_video.write(frame)


    output_video.release()
    cv2.destroyAllWindows()
