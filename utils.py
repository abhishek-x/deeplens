import cv2
import dlib
import numpy as np
from PIL import Image
from scipy import signal

def get_ppgs(vid_path):
    ppg_maps=[]
    # Define the input video file path
    video_path = vid_path

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) and total frame count
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Set the desired number of frames per segment
    frames_per_segment = 128  # Set to 128 frames per segment

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You need to download this file
    # detector.set_min_detection_confidence(0.5)
    # Initialize variables to keep track of segment number and frames
    segment_number = 1
    frame_count = 0
    ppgmap=np.empty([128,32,3])
    ind=0

    while True:
        #print("hi")
        ret, frame = cap.read()
        if not ret:
            break
            
        frame=cv2.resize(frame,(854,480))
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if(len(detector(gray))==0):
            continue

        face = detector(gray)[0]
        landmarks = predictor(gray, face)

        # Extract the coordinates of the nose (e.g., landmark point 30)
        nose_x = landmarks.part(30).x
        nose_y = landmarks.part(30).y

        # Define the ROI around the nose
        roi_size_width = 64
        roi_size_height = 32
        roi_x = nose_x - roi_size_width//2
        roi_y = nose_y - roi_size_height//2

        # Ensure ROI coordinates are within bounds
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_x_end = min(frame.shape[1], roi_x + roi_size_width)
        roi_y_end = min(frame.shape[0], roi_y + roi_size_height)

        # Extract the ROI
        roi = frame[roi_y:roi_y_end, roi_x:roi_x_end]
        
        # Calculate subregion size
        subregion_width = roi_size_width // 8  # 8 subregions horizontally
        subregion_height = roi_size_height // 4  # 4 subregions vertically

        subregions_r=np.zeros(32)
        subregions_y=np.zeros(32)
        subregions_v=np.zeros(32)
        k=0
        for i in range(4):
            for j in range(8):
                left = j * subregion_width
                upper = i * subregion_height
                right = (j + 1) * subregion_width
                lower = (i + 1) * subregion_height
                subregion=roi[upper:lower, left:right]

                roi_ycbcr = cv2.cvtColor(subregion, cv2.COLOR_BGR2YCrCb)
                roi_hsv = cv2.cvtColor(subregion, cv2.COLOR_BGR2HSV)
                
                y_comp = np.mean(roi_ycbcr[:, :, 0])  
                v_comp = np.mean(roi_hsv[:, :, 2])  
                rv=np.mean(subregion[:,:,0])
                
                subregions_r[k]=rv
                subregions_y[k]=y_comp
                subregions_v[k]=v_comp
                
                k+=1
                
        ppgmap[ind,:,0]=subregions_r
        ppgmap[ind,:,1]=subregions_y
        ppgmap[ind,:,2]=subregions_v
        #print(ppgmap[ind])
        ind+=1
        frame_count += 1
        #curr_seg.append(frame)
        #print(ppgmap[ind])
        if frame_count == frames_per_segment:
            min_values = np.min(ppgmap, axis=(0, 1))
            max_values = np.max(ppgmap, axis=(0, 1))
            scaled_data = ((ppgmap - min_values) / (max_values - min_values) * 255.0).astype(np.uint8)
            ppg_maps.append(scaled_data)
            segment_number += 1
            ppgmap=np.empty([128,32,3])
            ind=0
            frame_count = 0
    cap.release()
    return ppg_maps