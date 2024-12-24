# Car Counter Using YOLO and SORT

This project implements a car counting system using computer vision techniques, specifically leveraging the YOLOv8 (You Only Look Once) object detection model and the SORT (Simple Online and Realtime Tracking) algorithm. The system processes video input to detect, track, and count vehicles passing through a designated area.

## Features

- **Real-time Vehicle Detection**: Utilizes YOLOv8 for accurate car detection in video frames
- **Object Tracking**: Implements SORT algorithm for reliable vehicle tracking across frames
- **Automated Counting**: Maintains count of vehicles crossing a predefined line
- **Visual Feedback**: Displays processed video with bounding boxes, tracking IDs, and count statistics
- **Performance Optimization**: Uses region of interest masking to improve processing efficiency

## Prerequisites

Before running this project, ensure you have Python 3.x installed along with the following dependencies:

```bash
pip install opencv-python numpy ultralytics sort-tracker
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yigitcankzl/car-tracking-counting
cd car-tracking-counting
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the car counter with a video file:

```bash
python main.py --video_path path/to/your/video.mp4
```

### Command Line Arguments

- `--video_path`: Path to input video file (required)
- `--confidence`: Detection confidence threshold (default: 0.5)
- `--line_position`: Position of counting line (default: 0.6)
- `--show_display`: Enable/disable visual output (default: True)

### Example

```bash
python main.py --video_path traffic.mp4 
```

## Technical Details

### System Architecture

1. **Video Input Processing**
   - Reads video frames using OpenCV
   - Applies preprocessing for optimal detection

2. **Object Detection**
   - Utilizes YOLOv8 for vehicle detection
   - Filters detections based on confidence threshold
   - Applies region of interest masking

3. **Object Tracking**
   - SORT algorithm maintains object persistence
   - Assigns unique IDs to tracked vehicles
   - Handles occlusions and frame-to-frame tracking

4. **Counting Logic**
   - Monitors vehicle trajectories
   - Counts vehicles crossing the designated line
   - Prevents double-counting through ID tracking

### Output Visualization

The processed video display includes:
- Green bounding boxes around detected vehicles
- Unique tracking ID for each vehicle
- Red counting line
- Real-time vehicle count
- FPS (Frames Per Second) counter

## Performance Considerations

- Recommended minimum specifications:
  - CPU: Intel i5 or equivalent
  - RAM: 8GB
  - GPU: Optional but recommended for better performance
- Processing speed varies based on:
  - Input video resolution
  - Number of vehicles in frame
  - Hardware capabilities

## Troubleshooting

Common issues and solutions:

1. **Low FPS**
   - Reduce input video resolution
   - Adjust region of interest
   - Enable GPU acceleration if available

2. **Missed Detections**
   - Increase confidence threshold
   - Adjust lighting conditions
   - Verify video quality

3. **Double Counting**
   - Adjust line position
   - Modify tracking parameters
   - Check for video frame drops

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- SORT (Simple Online and Realtime Tracking) algorithm
- OpenCV community

## Contact

Yigitcan Kizil - kizilyigit33@hotmail.com
Project Link: [https://github.com/yigitcankzl/car-tracking-counting](https://github.com/yigitcankzl/car-tracking-counting)