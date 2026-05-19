# 🖼️ Vision Editor

A desktop image processing application built with Python and OpenCV, featuring real-time filter tools and a face detection comparison workflow.

---

## ✨ Features

### 🎨 Image Enhancement Tools
| Tool | Description |
|------|-------------|
| **Brightness** | Adjust pixel intensity (−255 to 255) |
| **Contrast** | Scale pixel values (alpha 0.1–3.0) |
| **Negative** | Invert all pixel values |
| **Grayscale** | Convert to grayscale (kept as 3-channel) |
| **Rotate 90°** | Quick 90-degree rotation |
| **Rotate Custom** | Rotate by any angle |
| **Zoom ×2 (Nearest)** | Zoom with nearest-neighbor interpolation |
| **Zoom ×2 (Bilinear)** | Zoom with bilinear interpolation |
| **Histogram Equalization** | Global contrast enhancement via YCrCb |
| **Gamma Correction** | Non-linear brightness adjustment |
| **CLAHE** | Adaptive local contrast enhancement |
| **Gaussian Blur** | Smooth noise with a Gaussian kernel |
| **Median Filter** | Remove salt-and-pepper noise |
| **Sobel Edge** | Gradient-based edge detection |
| **Canny Edge** | Two-threshold edge detection |

### 👁️ Face Detection (Before vs. After Workflow)
Upload a low-quality image → run detection → apply enhancements → run detection again → get a side-by-side comparison table showing faces found and processing time.

### 📊 Live Histograms
The dashboard shows intensity histograms for both the original and the processed output simultaneously.

---

## 🖥️ Dashboard Layout

```
┌─────────────────┬─────────────────────────────────────────┐
│   Controls      │              Dashboard                   │
│                 │  ┌─────────────┐  ┌──────────────────┐  │
│ [Upload Image]  │  │  Original   │  │ Input Histogram  │  │
│ [Save Output]   │  │             │  │                  │  │
│ [Reset]         │  └─────────────┘  └──────────────────┘  │
│                 │  ┌─────────────┐  ┌──────────────────┐  │
│ Select Tool:    │  │   Output    │  │ Output Histogram │  │
│ [Dropdown ▼]    │  │             │  │                  │  │
│ [Apply Tool]    │  └─────────────┘  └──────────────────┘  │
│                 │                                          │
│ [Face Detect]   │                                          │
└─────────────────┴─────────────────────────────────────────┘
```

---

## 📦 Requirements

```
opencv-python
numpy
Pillow
matplotlib
```

Install all dependencies:
```bash
pip install opencv-python numpy Pillow matplotlib
```

> `tkinter` is included with standard Python on Windows and macOS. On Linux you may need:
> ```bash
> sudo apt-get install python3-tk
> ```

---

## 🚀 Usage

```bash
python Vision_Editor.py
```

### Basic Workflow
1. Click **Upload Image** to load a JPG, PNG, or BMP file
2. Select a tool from the dropdown and click **Apply Tool**
3. Filters stack — each one applies on top of the previous output
4. Click **Save Output** to export the result
5. Click **Reset** to go back to the original image

### Face Detection Workflow (Before vs. After)
1. Upload a low-quality or poorly lit photo
2. Click **Run Face Detection** — captures the **BEFORE** score
3. Apply enhancement filters (e.g. CLAHE → Median Filter → Brightness)
4. Click **Run Face Detection** again — a comparison table appears:

```
========================================
       FACE DETECTION COMPARISON
========================================

                 BEFORE      AFTER
  Faces found  :  1           3
  Time (s)     :  0.0021      0.0018

────────────────────────────────────────
Conclusion:
Enhancement IMPROVED detection! More faces found.
========================================
```

---

## 🗂️ Project Structure

```
📦 vision-editor/
├── Vision_Editor.py    # Main application (single-file)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🔧 How It Works

- **GUI** — Built with `tkinter` and `ttk` for a native desktop look
- **Image processing** — All operations use OpenCV; the original image is never overwritten, filters chain on the current output
- **Face detection** — Haar Cascade classifier (`haarcascade_frontalface_default.xml`) bundled with OpenCV
- **Histograms** — Rendered inline using `matplotlib` embedded in tkinter via `FigureCanvasTkAgg`
