import base64
from datetime import datetime
import numpy as np
import streamlit as st
import PIL.Image
import cv2
from PIL import ImageOps
import dlib
from imutils import face_utils
import io
import pathlib


# helper function for facial landmark detection
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


# helper for downloading final image
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def blur_eyes(x, y, w, h, face, gray, color_copy):
    shape = dlib_facelandmark(gray, face)
    shape = face_utils.shape_to_np(shape)  # return coordinates of landmarks
    # lists to hold landmark coordinates for left and right eye
    left_eye = []
    right_eye = []
    i = 1
    for (a, b) in shape:
        # landmarks 37-42 are left eye
        if 36 < i < 43:
            left_eye.append((a, b))
        # landmarks 43-48 are right eye
        if 42 < i < 49:
            right_eye.append((a, b))
        i += 1

    # find ROIs of eyes plus padding
    height_constant = 4  # increase for bigger area blurred
    width_constant = 2  # increase for bigger area blurred
    left_eye_height = (left_eye[5][1] - left_eye[1][1]) * height_constant
    left_eye_width = (left_eye[3][0] - left_eye[0][0]) * width_constant
    left_eye_roi = color_copy[left_eye[1][1] - int(left_eye_height / 4): left_eye[1][1] -
                int(left_eye_height / 4) + left_eye_height, left_eye[0][0] - int(left_eye_width / 4):
                left_eye[0][0] - int(left_eye_width / 4) + left_eye_width]

    right_eye_height = (right_eye[5][1] - right_eye[1][1]) * height_constant
    right_eye_width = (right_eye[3][0] - right_eye[0][0]) * width_constant
    right_eye_roi = color_copy[right_eye[1][1] - int(right_eye_height / 4): right_eye[1][1] -
                int(right_eye_height / 4) + right_eye_height, right_eye[0][0] - int(right_eye_width / 4):
                right_eye[0][0] - int(right_eye_width / 4) + right_eye_width]

    left_eye_blurred = cv2.blur(left_eye_roi, (15, 15))
    right_eye_blurred = cv2.blur(right_eye_roi, (15, 15))

    color_copy[left_eye[1][1] - int(left_eye_height / 4): left_eye[1][1] - int(left_eye_height / 4) + left_eye_height,
                left_eye[0][0] - int(left_eye_width / 4): left_eye[0][0] - int(left_eye_width / 4) +
                left_eye_width] = left_eye_blurred
    color_copy[right_eye[1][1] - int(right_eye_height / 4): right_eye[1][1] - int(right_eye_height / 4) +
                right_eye_height, right_eye[0][0] - int(right_eye_width / 4): right_eye[0][0] - int(right_eye_width / 4)
                + right_eye_width] = right_eye_blurred
    return color_copy


def detect_faces(detector_name, img):

    # OpenCV DNN
    if detector_name == "OpenCV DNN":
        st.write("OpenCV DNN chosen")
        net = cv2.dnn.readNetFromCaffe(str(opencv_dnn_config_file), str(opencv_dnn_model_file))
        color_copy = np.array(img, dtype='uint8')
        color_copy2 = np.array(img, dtype='uint8')
        gray_image = ImageOps.grayscale(img)
        gray = np.array(gray_image, dtype='uint8')
        (h, w) = color_copy.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(color_copy, (300, 300)), 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        start = datetime.now()
        detections = net.forward()
        runtime = datetime.now() - start
        num_faces = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                cv2.rectangle(color_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
                num_faces += 1
        st.image(color_copy, caption=str(num_faces) + ' faces detected. Runtime: ' + str(runtime),
                 use_column_width=True)
        if st.button('Blur Eyes'):
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    w1 = x2 - x1
                    h1 = y2 - y1
                    face_rect = dlib.rectangle(x1, y1, int(x1 + w1), int(y1 + h1))
                    color_copy2 = blur_eyes(x1, y1, w, h, face_rect, gray, color_copy2)
            st.image(color_copy2, caption='Blurred')

    # haar-cascade
    if detector_name == "OpenCV Haar-Cascade":
        st.write("OpenCV Haar-Cascade chosen")
        face_cascade = cv2.CascadeClassifier(
            'C:/Users/hanna/PycharmProjects/EyeBlur/haarcascade_frontalface_default.xml')
        color_copy = np.array(img, dtype='uint8')
        color_copy2 = np.array(img, dtype='uint8')
        gray_image = ImageOps.grayscale(img)
        gray = np.array(gray_image, dtype='uint8')
        start = datetime.now()
        faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
        runtime = datetime.now() - start
        num_faces = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(color_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            num_faces += 1
        st.image(color_copy, caption=str(num_faces) + ' faces detected. Runtime: ' + str(runtime),
                 use_column_width=True)
        if st.button('Blur Eyes'):
            for (x, y, w, h) in faces:
                face_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                color_copy2 = blur_eyes(x, y, w, h, face_rect, gray, color_copy2)
            st.image(color_copy2, caption='Blurred')

    # Dlib HoG
    if detector_name == "Dlib HoG":
        st.write("Dlib HoG chosen")
        hog_face_detector = dlib.get_frontal_face_detector()
        color_copy = np.array(img, dtype='uint8')
        color_copy2 = np.array(img, dtype='uint8')
        gray_image = ImageOps.grayscale(img)
        gray = np.array(gray_image, dtype='uint8')
        num_faces = 0
        start = datetime.now()
        faces = hog_face_detector(gray, upsample)
        runtime = datetime.now() - start
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            cv2.rectangle(color_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            num_faces += 1
        st.image(color_copy, caption=str(num_faces) + ' faces detected. Runtime: ' + str(runtime))
        if st.button('Blur Eyes'):
            for face in faces:
                color_copy2 = blur_eyes(x, y, w, h, face, gray, color_copy2)
            st.image(color_copy2, caption='Blurred')

    # Dlib CNN
    if detector_name == "Dlib CNN (slow)":
        st.write("Dlib CNN chosen")
        cnn_face_detector = dlib.cnn_face_detection_model_v1(str(dlib_cnn_face_detector))
        color_copy = np.array(img, dtype='uint8')
        color_copy2 = np.array(img, dtype='uint8')
        gray_image = ImageOps.grayscale(img)
        gray = np.array(gray_image, dtype='uint8')
        num_faces = 0
        start = datetime.now()
        faces = cnn_face_detector(gray, upsample)
        runtime = datetime.now() - start
        for face in faces:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
            cv2.rectangle(color_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            num_faces += 1
        st.image(color_copy, caption=str(num_faces) + ' faces detected. Runtime: ' + str(runtime),
                 use_column_width=True)
        if st.button('Blur Eyes'):
            for face in faces:
                face_rect = face.rect
                color_copy2 = blur_eyes(x, y, w, h, face_rect, gray, color_copy2)
            st.image(color_copy2, caption='Blurred')
    return color_copy2


# ---------------------Start of program--------------------------

# relative file paths
directory = pathlib.Path(pathlib.Path(__file__).parent)
dlib_cnn_face_detector = directory / 'mmod_human_face_detector.dat'
haar_cascade_face_detector = directory / 'haarcascade_frontalface_default.xml'
opencv_dnn_model_file = directory / 'res10_300x300_ssd_iter_140000.caffemodel'
opencv_dnn_config_file = directory / 'deploy.prototxt.txt'
facelandmark = directory / 'shape_predictor_68_face_landmarks.dat'

# default values for sliders
upsample = 0
min_neighbors = 4
scale_factor = 1.05
conf_threshold = .9

# face landmarks used for finding eyes
dlib_facelandmark = dlib.shape_predictor(str(facelandmark))

# main hub of app
st.title("Anonymize Faces")
with st.expander("About this App"):
    st.write("""
        This app allows the user to upload an image and anonymize (blur eyes) each face present in 
        6 steps:\n
        1. Upload image using the 'Browse Files' button.
        2. Click the arrow on the left of the screen to expand the sidebar
        3. Using the dropdown menu on the left, choose the desired face detector.
        4. Using the sliders below the dropdown menu, adjust parameters until all faces are found.
        5. Click the 'Blur Eyes' button below the image.
        6. Download the anonymized image by clicking the blue 'Download Image' link below the 'Blur Eyes' button.
        WARNING: Dlib CNN face detector is very slow. Increasing the upsample value may cause runtime to take minutes.
    """)
uploaded_file = st.file_uploader("Choose an Image")
detector_name = st.sidebar.selectbox("Select Face Detector",
                                     ("OpenCV DNN", "OpenCV Haar-Cascade", "Dlib HoG", "Dlib CNN (slow)"))

# create sidebar menu
if detector_name == "Dlib HoG":
    upsample = st.sidebar.slider("Select Upsample Value", 0, 5, 1)
if detector_name == "Dlib CNN (slow)":
    upsample = st.sidebar.slider("Select Upsample Value (increasing may cause detection to take minutes)", 0, 2, 0)
if detector_name == "OpenCV Haar-Cascade":
    min_neighbors = st.sidebar.slider("Select Min. Neighbors", 1, 6, 4)
    scale_factor = st.sidebar.slider("Select Scale Factor", 1.01, 1.40, 1.05)
if detector_name == "OpenCV DNN":
    conf_threshold = st.sidebar.slider("Select Confidence Threshold", .10, .99, .85)

# detect faces once an image is uploaded
if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    blurred = detect_faces(detector_name, image)
    # download link for final image
    result = PIL.Image.fromarray(blurred)
    st.markdown(get_image_download_link(result, 'blurred_eyes.jpg', 'Download Image'), unsafe_allow_html=True)
