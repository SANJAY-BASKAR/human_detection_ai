import torch
import cv2
import argparse


# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# Function to detect people using YOLOv5
# Function to detect people using YOLOv5
def detect_people_yolo(frame):
    results = model(frame)  # Run detection
    frame = results.render()[0]  # Render the frame with detection boxes

    # Make a copy of the frame to avoid modifying the original frame
    frame_copy = frame.copy()

    # Get the number of people detected (class 0 is 'person')
    count = sum([1 for *box, conf, cls in results.xyxy[0] if int(cls) == 0])

    # Display total count of persons detected on the frame copy
    cv2.putText(frame_copy, f'Total Persons: {count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame_copy


# Function to handle webcam feed
def detectByCamera():
    cap = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame if necessary
        frame = cv2.resize(frame, (800, 600))

        # Detect people in the frame
        frame = detect_people_yolo(frame)

        # Show the frame
        cv2.imshow('YOLOv5 Person Detection', frame)

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to handle video file input
def detectByPathVideo(path):
    cap = cv2.VideoCapture(path)
    print('Detecting people...')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame if necessary
        frame = cv2.resize(frame, (800, 600))

        # Detect people in the frame
        frame = detect_people_yolo(frame)

        # Show the frame
        cv2.imshow('YOLOv5 Person Detection', frame)

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to handle image input
def detectByPathImage(path, output_path=None):
    image = cv2.imread(path)
    image = cv2.resize(image, (800, 600))

    # Detect people in the image
    result_image = detect_people_yolo(image)

    # Display result image
    cv2.imshow('YOLOv5 Person Detection', result_image)

    # Save the result image if an output path is provided
    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Main function to handle arguments and process input
def humanDetector(args):
    image_path = args["image"]
    video_path = args["video"]
    camera = args["camera"]

    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera()
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args["output"])


# Argument parser to handle command-line inputs
def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="Path to video file")
    arg_parse.add_argument("-i", "--image", default=None, help="Path to image file")
    arg_parse.add_argument("-c", "--camera", action='store_true', help="Use webcam for real-time detection")
    arg_parse.add_argument("-o", "--output", type=str, help="Path to output video/image file")
    args = vars(arg_parse.parse_args())
    return args


if __name__ == "__main__":
    args = argsParser()
    humanDetector(args)
