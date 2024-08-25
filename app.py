import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import cv2
import dlib
import numpy as np
from PIL import Image
from scipy import signal
from keras.optimizers import Adam
from keras.models import load_model
from collections import Counter
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings("ignore")

# Load your pre-trained model
custom_optimizer = Adam(learning_rate=0.0005)
model = load_model("../models/model.h5", compile=False)
model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

def detect_blur(frame, threshold=100):
    return cv2.Laplacian(frame, cv2.CV_64F).var() < threshold

def plot_color_channel_means(means, title, color_labels, colors):
    plt.figure(figsize=(10, 4))
    n_channels = len(color_labels)
    for i, (color_label, color) in enumerate(zip(color_labels, colors)):
        plt.plot(means[i::n_channels], label=f'{color_label} channel', color=color, marker='o', linestyle='-', linewidth=1, markersize=2)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Mean Intensity', fontsize=12)
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    return plt

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
    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")  # You need to download this file
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

# Streamlit interface
st.markdown("<h1 style='text-align: center;'>DeepLens: Detect Synthetic Facial Manipulations in Videos</h1>", unsafe_allow_html=True)

# Custom CSS for the hero section
hero_section_css = """
<style>
.hero-container {
    text-align: center;
    margin: 50px 0;
}
.hero-title {
    font-size: 40px;
    font-weight: bold;
    color: #0078ff;  /* Change the color as per your theme */
}
.hero-subtitle {
    font-size: 24px;
    color: #fff;  /* Change the color as per your theme */
    margin-top: 10px;
}
</style>
"""

# Add the custom CSS to the app
st.markdown(hero_section_css, unsafe_allow_html=True)

# Hero section content
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Discover the Unseen in Every Frame</div>
    <div class="hero-subtitle">Your Video, Decoded and Demystified</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.card {
    border-radius: 10px;
    box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.2);
    padding: 20px;
    margin: 10px 0;
    background-color: #fff;
}
.title {
    font-size: 18px;
    font-weight: bold;
    color: #0078ff;
}
.description {
    font-size: 14px;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# Layout for the feature cards
col1, col2, col3 = st.columns(3)

# Frame-by-Frame Analysis Card
with col1:
    st.markdown("""
    <div class="card">
        <div class="title">Frame-by-Frame<br>Analysis</div>
        <div class="description">
            Dive into every detail of each frame, uncovering hidden patterns and insights.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Color Channel Analysis Card
with col2:
    st.markdown("""
    <div class="card">
        <div class="title">Color Channel<br>Analysis</div>
        <div class="description">
            Explore the nuances of color variations and their impact on your video's storytelling.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Authenticity Verification Card
with col3:
    st.markdown("""
    <div class="card">
        <div class="title">Facial Manipulation<br>Detection</div>
        <div class="description">
            Determine the originality of your content, differentiating the real from the altered.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.header("")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])

if uploaded_video is not None:
    # Process the video and make predictions
    st.subheader('Uploading the video')
    temp_file = "temp_video.mp4"
    with open(temp_file, 'wb') as f:
        f.write(uploaded_video.read())
    # Display video
    st.video(temp_file)

    if st.button('Analyze Video'):
            cap = cv2.VideoCapture(temp_file)
            st.title("Facial Manipulation Detection Started")
            st.write("Detecting Facial Landmarks...")
            ppgs_for_video = get_ppgs(temp_file)
            st.write("Creating Photoplethysmographic Maps...")
            segment_predictions = []
            confidence_scores = []  # To store confidence scores
            for k in ppgs_for_video:
                k = k / 255.0
                batch_of_images = np.array([k] * 32)
                predictions = model.predict([batch_of_images])
                confidence = predictions[0][0]  # Get confidence score
                confidence_scores.append(confidence)  # Store confidence score
                pc = 1 if confidence > 0.5 else 0
                segment_predictions.append(pc)
            st.write("Fetching results from model...")

            vid_prediction = "Manipulated" if Counter(segment_predictions)[0] >= Counter(segment_predictions)[1] else "Original"
            st.subheader("Classification Results")
            if vid_prediction == "Original":
                st.success(f"✅ Prediction from our model: {vid_prediction}")
            else:
                st.error(f"❌ Prediction from our model: {vid_prediction}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_count = 0
            prev_frame = None
            ssim_values = []
            frame_times = []
            blur_count = 0
            color_distribution = np.zeros((3,), dtype=np.float64)
            rgb_means, hsv_means, ycrcb_means = [], [], []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frame_count += 1

                if detect_blur(frame):
                    blur_count += 1

                if prev_frame is not None:
                    # Calculate SSIM
                    score, _ = ssim(prev_frame, gray, full=True)
                    ssim_values.append(score)
                    frame_times.append(frame_count / fps)  # Convert frame number to time

                prev_frame = gray

                # Analyze RGB Channels
                for i in range(3):
                    rgb_means.append(np.mean(frame[:, :, i]))

                # Convert and Analyze HSV Channels
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                for i in range(3):
                    hsv_means.append(np.mean(hsv_frame[:, :, i]))

                # Convert and Analyze YCrCb Channels
                ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                for i in range(3):
                    ycrcb_means.append(np.mean(ycrcb_frame[:, :, i]))

                # Basic color analysis (summing up all the pixel values for each color channel)
                color_distribution += np.sum(frame, axis=(0, 1))

            
            cap.release()
            average_color = color_distribution / (frame_count * width * height)

            st.subheader('Video Analysis Report')

            st.markdown("""
                <style>
                    .report-box {
                        border: 1px solid #aaa;
                        border-radius: 10px;
                        padding: 10px;
                        margin: 10px 0;
                    }
                    .report-label {
                        font-weight: bold;
                    }
                </style>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="report-box">
                    <p><span class="report-label">FPS:</span> {fps}</p>
                    <p><span class="report-label">Total Frames:</span> {total_frames}</p>
                    <p><span class="report-label">Duration (seconds):</span> {duration:.2f}</p>
                    <p><span class="report-label">Resolution:</span> {width}x{height}</p>
                    <p><span class="report-label">Blurry Frames Detected:</span> {blur_count} out of {total_frames}</p>
                    <p><span class="report-label">Average Color Distribution (RGB):</span> {average_color}</p>
                    <p><span class="report-label">Facial Manipulation:</span> {vid_prediction}</p>
                </div>
            """, unsafe_allow_html=True)

            plt.figure(figsize=(10, 4))
            plt.plot(frame_times, ssim_values, color='blue', marker='o', linestyle='-', linewidth=1, markersize=5)
            plt.title('Structural Similarity Index (SSIM) Over Video Frames')
            plt.xlabel('Duration (in seconds)')
            plt.ylabel('SSIM Value')
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt)

            st.subheader("Color Channel Analysis")
            # Plot RGB Analysis
            rgb_colors = ['red', 'green', 'blue']
            rgb_plot = plot_color_channel_means(rgb_means, 'RGB Color Channel Analysis', ['Red', 'Green', 'Blue'], rgb_colors)
            st.pyplot(rgb_plot)

            # Plot HSV Analysis
            hsv_colors = ['gold', 'cyan', 'magenta']
            hsv_plot = plot_color_channel_means(hsv_means, 'HSV Color Channel Analysis', ['Hue', 'Saturation', 'Value'], hsv_colors)
            st.pyplot(hsv_plot)

            # Plot YCrCb Analysis
            ycrcb_colors = ['black', 'brown', 'blueviolet']
            ycrcb_plot = plot_color_channel_means(ycrcb_means, 'YCrCb Color Channel Analysis', ['Luma (Y)', 'Chroma (Cr)', 'Chroma (Cb)'], ycrcb_colors)
            st.pyplot(ycrcb_plot)

footer_html = """
<style>
.footer {
    font-family: Arial, sans-serif;
    font-size: 14px;
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    background-color: #f1f1f1;
    color: #666;
    border-top: 1px solid #e6e6e6;
}
.footer a {
    color: #008cff;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
<div class="footer">
    <p>Made with ❤️ by <a href="https://github.com/abhishek-x" target="_blank">Abhishek Aggarwal</a>. © 2023 All Rights Reserved.</p>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)