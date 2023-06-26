# face-recognition
simple face recognition ML project with training program and model testing program
# Face Recognition Program

This is a simple face recognition program that uses convolutional neural networks (CNNs) to recognize faces in real-time. The program is implemented using TensorFlow and OpenCV.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

The face recognition program utilizes a trained CNN model to perform real-time face recognition using a webcam or video stream. The program detects faces in each frame, preprocesses the face region, and feeds it to the model for prediction. The predicted class name and confidence score are then displayed on the frame along with a rectangle around the detected face.

## Features

- Real-time face recognition using a webcam or video stream.
- Automatic face detection using Haar cascades.
- Display of predicted class names and confidence scores on the video frames.

## Installation

1. Clone the repository:

  git clone https://github.com/viveksethaliya/face-recognition/


2. Install the required dependencies:

  pip install -r requirements.txt

3. Download the pre-trained model weights and place them in the project directory.

4. Prepare your dataset for training and validation by following the instructions in the [Dataset](#dataset) section.

## Usage

1. Connect a webcam or provide a video stream as an input source.

2. Run the face recognition program:

3. The program will open a window showing the real-time video stream with face recognition results.

4. Press 'q' to quit the program.

## Dataset

To train the face recognition model, you need to prepare a dataset of images for each individual you want to recognize. The dataset should be organized into separate directories, with each directory containing images of a single individual.

The structure of the dataset directory should be as follows:

dataset/
├── train dataset/
│ ├── class1/
│ │ ├── image1.jpg
│ │ ├── image2.jpg
│ │ └── ...
│ ├── class2/
│ │ ├── image1.jpg
│ │ ├── image2.jpg
│ │ └── ...
│ ├── ...
│ 
├── validation dataset/
│ ├── class1/
│ │ ├── image1.jpg
│ │ ├── image2.jpg
│ │ └── ...
│ ├── class2/
│ │ ├── image1.jpg
│ │ ├── image2.jpg
│ │ └── ...
│ ├── ...

Make sure to resize and preprocess the images to the desired input size before training the model.

## Model Training

To train your own face recognition model, follow these steps:

1. Prepare your dataset as explained in the [Dataset](#dataset) section.

2. Configure the hyperparameters in the training script.

3. Run the training script:

4. The trained model will be saved as 'model.h5' or a specified file.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Haar cascades](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
