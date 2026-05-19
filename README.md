# Vision Editor

Vision Editor is a Python-based Image Processing and Computer Vision application built using Tkinter and OpenCV.  
The project provides an interactive GUI that allows users to apply image enhancement techniques, visualize histograms, and evaluate how preprocessing affects face detection accuracy.

---

## Features

### Image Processing Operations
- Brightness Adjustment
- Contrast Adjustment
- Negative Transformation
- Grayscale Conversion

### Geometric Transformations
- Rotate Image (90° & Custom Angle)
- Zoom using:
  - Nearest Neighbor Interpolation
  - Bilinear Interpolation

### Enhancement Techniques
- Histogram Equalization
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gamma Correction

### Filtering Operations
- Gaussian Blur
- Median Filter

### Edge Detection
- Sobel Edge Detection
- Canny Edge Detection

### Computer Vision Task
- Face Detection using Haar Cascade Classifier
- Before vs After comparison workflow
- Detection accuracy evaluation on enhanced images

### Visualization
- Original and Processed Image Display
- Histogram Visualization
- Real-time GUI Updates

---

## Technologies Used

- Python
- OpenCV
- Tkinter
- NumPy
- Matplotlib
- Pillow (PIL)

---

## Project Objective

The goal of this project is to demonstrate how image preprocessing techniques can improve the performance of Computer Vision tasks.

The application follows the principle:

> Garbage In = Garbage Out

Low-quality images often reduce AI model accuracy.  
This project shows how enhancement techniques such as CLAHE, Histogram Equalization, and Filtering can improve face detection results.

---

## Workflow

1. Upload a low-quality image
2. Run Face Detection (Baseline Result)
3. Apply enhancement filters
4. Run Face Detection again
5. Compare BEFORE vs AFTER results
