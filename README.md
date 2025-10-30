# DeepFace - Face Recognition and Analysis

A Python-based face recognition and analysis system using the DeepFace library. This project demonstrates face verification, facial attribute analysis, and face identification tasks.

## Features

- **Face Verification**: Verify if two images contain the same person
- **Facial Analysis**: Analyze age, gender, and emotion from face images
- **Face Identification**: Identify and match faces against a database of known faces
- **Face Detection**: Detect and annotate facial regions in images
- **Support for Multiple Backends**: Works with OpenCV and RetinaFace detectors

## Project Structure

```
Deepface/
├── DeepFace.ipynb          # Main Jupyter notebook with all examples
├── face_db/                # Face database for identification
│   ├── MESSI/              # Training images for Messi
│   ├── Ronaldo/            # Training images for Ronaldo
│   └── musk/               # Training images for Elon Musk
├── MESSI.jpg               # Sample test image
├── Ronaldo.jpg             # Sample test image
├── musk.jpg                # Sample test image
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Install Required Dependencies

```bash
pip install deepface opencv-python matplotlib numpy pandas
```

### Optional Dependencies
For better face detection performance:
```bash
pip install tensorflow keras
```

## Usage

### 1. Face Verification

Verify if two face images belong to the same person:

```python
from deepface import DeepFace
import cv2

# Load images
img1 = cv2.imread('MESSI.jpg')
img2 = cv2.imread('Ronaldo.jpg')

# Verify
result = DeepFace.verify(img1, img2)
print(f"Same person? {result['verified']}")
print(f"Distance: {result['distance']:.2f}")
```

### 2. Facial Analysis

Analyze facial attributes (age, gender, emotion):

```python
from deepface import DeepFace
import cv2

img = cv2.imread('MESSI.jpg')
analysis = DeepFace.analyze(img, actions=['age', 'gender', 'emotion'])

print(f"Age: {analysis[0]['age']}")
print(f"Gender: {analysis[0]['dominant_gender']}")
print(f"Emotion: {analysis[0]['dominant_emotion']}")
```

### 3. Face Identification

Identify a face by finding the best match in a database:

```python
from deepface import DeepFace
import cv2

img = cv2.imread('MESSI.jpg')

# Find matching faces in database
matches = DeepFace.find(img_path=img, db_path='face_db', enforce_detection=False)

if matches and len(matches[0]) > 0:
    best_match = matches[0].iloc[0]
    print(f"Identified as: {best_match['identity']}")
    print(f"Confidence: {(1 - best_match['distance']) * 100:.1f}%")
```

### 4. Face Detection and Annotation

Detect faces and annotate them with identity information:

```python
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('musk.jpg')

# Extract faces with bounding boxes
face_objs = DeepFace.extract_faces(img_path=img, detector_backend='retinaface', enforce_detection=False)

# Find matches in database
dfs = DeepFace.find(img_path=img, db_path='face_db', enforce_detection=False, detector_backend='retinaface', silent=True)

# Draw rectangles and labels
for result in dfs:
    if len(result) > 0:
        best_match = result.iloc[0]
        identity = best_match['identity'].split('/')[-2]
        confidence = (1 - best_match['distance']) * 100
        # Draw bounding box and label
        # ... (see notebook for complete implementation)
```

## Building a Face Database

To add new faces to the database for identification:

```python
import os
import shutil

# Create directories for each person
identities = ["Person1", "Person2", "Person3"]
db_path = "face_db"

for identity in identities:
    identity_path = os.path.join(db_path, identity)
    if not os.path.exists(identity_path):
        os.makedirs(identity_path)
        print(f"Created directory: {identity_path}")

# Add face images to respective directories
# face_db/
#   ├── Person1/
#   │   ├── image1.jpg
#   │   ├── image2.jpg
#   └── Person2/
#       ├── image1.jpg
#       └── image2.jpg
```

## Model and Detector Backends

### Available Models
- **VGG-Face** (default): Fast and accurate
- **ArcFace**: State-of-the-art accuracy
- **DeepID**: Alternative model
- **FaceNet**: Google's model

### Available Detectors
- **opencv** (default): Fast CPU-based detection
- **retinaface**: More accurate, handles various angles
- **mtcnn**: Multi-task cascaded convolutional networks

## Dependencies

The project relies on several key libraries:

- **deepface**: Face recognition and analysis
- **opencv-python**: Image processing
- **tensorflow**: Deep learning backend
- **keras**: Neural network API
- **matplotlib**: Visualization
- **numpy**: Numerical computations
- **pandas**: Data manipulation

See the notebook for the complete installation output.

## Example Output

### Face Verification
```
Verification Result: {'verified': False, 'distance': 0.955075, 'threshold': 0.68, 'confidence': 0.69, 'model': 'VGG-Face', 'detector_backend': 'opencv', 'similarity_metric': 'cosine'}
```

### Facial Analysis
```
Analysis Results:
Age: 26
Gender: Man
Emotion: neutral
```

### Face Identification
```
Best match: MESSI (Distance: 0.40)
Identified as: MESSI
Confidence: 60.0%
```

## Running the Notebook

To run the complete project with all examples:

```bash
jupyter notebook DeepFace.ipynb
```

Then execute the cells in sequence to:
1. Install dependencies
2. Create face database directories
3. Run face verification examples
4. Perform facial analysis
5. Identify faces from the database
6. Visualize results with annotations

## Tips and Best Practices

1. **Image Quality**: Use clear, well-lit face images for best results
2. **Face Database**: Include multiple images per person (3-5 recommended) for better accuracy
3. **Distance Threshold**: Lower distances indicate higher confidence in matches (typically < 0.6 for matches)
4. **Detector Backend**: Use 'retinaface' for better accuracy with challenging images
5. **Batch Processing**: For multiple images, consider optimizing with batch processing

## Troubleshooting

### No faces detected
- Ensure images have clear, visible faces
- Try different `detector_backend` values
- Lower or disable `enforce_detection` parameter

### Low identification accuracy
- Add more diverse images to the database
- Ensure consistent lighting in database images
- Try different model backends (e.g., 'ArcFace' instead of 'VGG-Face')

## License

This project is created for educational and demonstration purposes. The DeepFace library is licensed under the MIT License. Please refer to the original DeepFace repository for more information.

## References

- [DeepFace GitHub](https://github.com/serengp/deepface)
- [Face Recognition Research](https://arxiv.org/abs/1503.03832)

## Notes

- The notebook includes test images of public figures (Messi, Ronaldo, Musk) for demonstration purposes only
- Face identification accuracy depends on image quality and database size
- Always ensure compliance with privacy regulations when using facial recognition
