import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

model_path = r'C:/Users/finda/OneDrive/Desktop/procode/AI-ML/Drone simulation/NEW/collectedimages copy'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

dataset_dir = r'C:/Users/finda/OneDrive/Desktop/procode/AI-ML/Drone simulation/NEW/collectedimages copy'
os.makedirs(dataset_dir, exist_ok=True)

def save_landmarks(image, landmarks, count):
    image_path = os.path.join(dataset_dir, f'image_{count}.jpg')
    cv2.imwrite(image_path, image)
    landmarks_path = os.path.join(dataset_dir, f'landmarks_{count}.npy')
    np.save(landmarks_path, landmarks)

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global capture_count
    if result.hand_landmarks:
        landmarks = np.array(result.hand_landmarks[0].landmark)
        save_landmarks(output_image.numpy_view(), landmarks, capture_count)
        capture_count += 1

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

capture_count = 0
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image, width=image.shape[1], height=image.shape[0])
        landmarker.detect_async(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

        cv2.imshow('Hand Landmarker', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def load_dataset(dataset_dir):
    images = []
    landmarks = []
    for filename in os.listdir(dataset_dir):
        if filename.startswith('d'):
            image_path = os.path.join(dataset_dir, filename)
            landmarks_path = os.path.join(dataset_dir, filename.replace('d ','landmarks_').replace('.jpg', '.npy'))
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            images.append(image)
            landmarks.append(np.load(landmarks_path))
    return np.array(images), np.array(landmarks)

train_images, train_landmarks = load_dataset(dataset_dir)

def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(42))  
    return model

image_height, image_width = 224, 224  
input_shape = (image_height, image_width, 3)
model = create_model(input_shape)
model.compile(optimizer='adam', loss='mean_squared_error')

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_landmarks)).batch(32)

num_epochs = 10
model.fit(train_dataset, epochs=num_epochs)

def predict_gesture(image):
    image = tf.expand_dims(image, axis=0)  
    output = model.predict(image)
    landmarks = output.reshape(-1, 21, 2)
    gesture = recognize_gesture(landmarks)
    return gesture

def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    if thumb_tip[0] < index_tip[0] and thumb_tip[0] < middle_tip[0]:
        return "Left"
    elif thumb_tip[0] > index_tip[0] and thumb_tip[0] > middle_tip[0]:
        return "Right"
    elif thumb_tip[1] < index_tip[1] and thumb_tip[1] < middle_tip[1]:
        return "Up"
    elif thumb_tip[1] > index_tip[1] and thumb_tip[1] > middle_tip[1]:
        return "Down"
    elif index_tip[1] < middle_tip[1] and ring_tip[1] < pinky_tip[1]:
        return "Forward"
    elif index_tip[1] > middle_tip[1] and ring_tip[1] > pinky_tip[1]:
        return "Backward"
    else:
        return "Unknown Gesture"

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        landmarker.detect_async(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

        gesture = predict_gesture(image)
        cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Landmarker', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()