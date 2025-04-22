

# YOLOv5 Human Detection

This project utilizes the YOLOv5 (You Only Look Once version 5) model to detect human beings in images, videos, and real-time webcam feed. It leverages PyTorch and OpenCV for object detection and visualization.

## Features

- **Real-time Webcam Detection**: Detect people using your webcam.
- **Image Detection**: Detect people in static images.
- **Video Detection**: Detect people in video files.
- **Detection Count**: Displays the total number of people detected in each frame.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- YOLOv5 pre-trained model

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install opencv-python
```

## Usage

You can use this script with command-line arguments to process images, videos, or webcam input.

### Command-line Arguments

- `-v, --video`: Path to the video file for processing.
- `-i, --image`: Path to the image file for processing.
- `-c, --camera`: Flag to use the webcam for real-time detection.
- `-o, --output`: Path to save the output (image or video) after detection.

### Examples

#### 1. Webcam Real-Time Detection

```bash
python detect_people.py --camera
```

#### 2. Image Detection

```bash
python detect_people.py --image path/to/your/image.jpg --output path/to/output.jpg
```

#### 3. Video Detection

```bash
python detect_people.py --video path/to/your/video.mp4 --output path/to/output_video.mp4
```

## How it Works

1. The script loads the pre-trained YOLOv5 model using PyTorch.
2. Based on the user input, the model processes either a webcam feed, video, or image file.
3. For each frame, the model detects objects and counts how many people (class 0) are present.
4. The number of detected people is displayed on the image/video feed.
5. If an output path is provided, the result is saved.

## Contributing

Feel free to open issues, fork the project, and create pull requests. Your contributions are welcome!

---

Let me know if you'd like to make any edits or additions!
