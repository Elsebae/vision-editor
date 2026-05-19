"""
Vision Editor - Complete Single File
Workflow for Task 2:
  1. Upload a bad quality photo
  2. Click "Run Face Detection" → get score (this is the BEFORE score)
  3. Apply filters manually to enhance the image
  4. Click "Run Face Detection" again → get comparison table (BEFORE vs AFTER)
"""

import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =============================================================================
# IMAGE OPERATIONS
# =============================================================================

def brightness(img, value=50):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=value)

def contrast(img, alpha=1.5):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def negative(img):
    return cv2.bitwise_not(img)

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def rotate(img, angle=90):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def zoom(img, scale=2.0, method=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=method)
    # Center-crop back to original size
    rh, rw = resized.shape[:2]
    y1 = (rh - h) // 2
    x1 = (rw - w) // 2
    return resized[y1:y1+h, x1:x1+w]

def histogram_equalization(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def gamma_correction(img, gamma=0.7):
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)

def gaussian_blur(img, k=5):
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(img, (k, k), 0)

def median_filter(img, k=5):
    k = k if k % 2 == 1 else k + 1
    return cv2.medianBlur(img, k)

def clahe(img):
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = clahe_obj.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def sobel_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edges = cv2.normalize(cv2.magnitude(sobelx, sobely), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def canny_edge(img, t1=100, t2=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, t1, t2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# =============================================================================
# FACE DETECTION HELPER
# Returns (annotated image, num faces, elapsed time)
# =============================================================================

def detect_faces(img):
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.perf_counter()
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    elapsed = time.perf_counter() - start

    result = img.copy()
    num_faces = len(faces) if len(faces) > 0 else 0
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return result, num_faces, elapsed


# =============================================================================
# GUI
# =============================================================================

class VisionEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision Editor")
        self.root.geometry("1200x700")

        self.original = None   # the uploaded image (never changed by filters)
        self.output   = None   # result after applying filters

        # Store the BEFORE detection result to compare later
        self.before_faces = None
        self.before_time  = None
        self.before_img   = None  # annotated before image

        self._build_ui()

    # ---------------------------------------------------------------
    # BUILD UI
    # ---------------------------------------------------------------

    def _build_ui(self):
        # Left: controls
        left = ttk.LabelFrame(self.root, text="Controls", padding=10)
        left.grid(row=0, column=0, sticky="ns", padx=8, pady=8)

        # Right: dashboard
        right = ttk.LabelFrame(self.root, text="Dashboard", padding=8)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        right.columnconfigure((0, 1), weight=1)
        right.rowconfigure((0, 1), weight=1)

        # --- Buttons ---
        ttk.Button(left, text="Upload Image", command=self.load_image).pack(fill="x", pady=4)
        ttk.Button(left, text="Save Output",  command=self.save_output).pack(fill="x", pady=4)
        ttk.Button(left, text="Reset",        command=self.reset).pack(fill="x", pady=4)
        ttk.Separator(left).pack(fill="x", pady=8)

        # --- Tool selector ---
        ttk.Label(left, text="Select Tool:").pack(anchor="w")
        self.tool = ttk.Combobox(left, state="readonly", width=28, values=[
            "Brightness", "Contrast", "Negative", "Grayscale",
            "Rotate 90°", "Rotate Custom",
            "Zoom x2 (Nearest)", "Zoom x2 (Bilinear)",
            "Histogram Equalization", "Gamma Correction", "CLAHE",
            "Gaussian Blur", "Median Filter",
            "Sobel Edge", "Canny Edge",
        ])
        self.tool.current(0)
        self.tool.pack(fill="x", pady=4)

        ttk.Button(left, text="Apply Tool", command=self.apply_tool).pack(fill="x", pady=6)

        ttk.Separator(left).pack(fill="x", pady=8)

        # --- Face Detection button (Task 2) ---
        ttk.Label(left, text="Task 2 - Face Detection:").pack(anchor="w")
        ttk.Button(left, text="Run Face Detection", command=self.run_face_detection,).pack(fill="x", pady=4)

        # --- Status ---
        self.status = ttk.Label(left, text="Load an image to start.", wraplength=220)
        self.status.pack(fill="x", pady=10)

        # --- Four display panels ---
        for title, r, c in [("Original", 0, 0), ("Input Histogram", 0, 1),
                             ("Output",   1, 0), ("Output Histogram", 1, 1)]:
            f = ttk.LabelFrame(right, text=title, padding=4)
            f.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            f.columnconfigure(0, weight=1)
            f.rowconfigure(0, weight=1)

            if title == "Original":
                self.lbl_original = ttk.Label(f, text="No image", anchor="center")
                self.lbl_original.grid(sticky="nsew")
            elif title == "Output":
                self.lbl_output = ttk.Label(f, text="No output", anchor="center")
                self.lbl_output.grid(sticky="nsew")
            elif title == "Input Histogram":
                self.fig_in = Figure(figsize=(4, 2.5), dpi=90)
                self.ax_in  = self.fig_in.add_subplot(111)
                self.cv_in  = FigureCanvasTkAgg(self.fig_in, master=f)
                self.cv_in.get_tk_widget().grid(sticky="nsew")
            else:
                self.fig_out = Figure(figsize=(4, 2.5), dpi=90)
                self.ax_out  = self.fig_out.add_subplot(111)
                self.cv_out  = FigureCanvasTkAgg(self.fig_out, master=f)
                self.cv_out.get_tk_widget().grid(sticky="nsew")

    # ---------------------------------------------------------------
    # LOAD / SAVE / RESET
    # ---------------------------------------------------------------

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not open image.")
            return
        self.original     = img
        self.output       = img.copy()
        self.before_faces = None   # reset detection history on new image
        self.before_time  = None
        self.before_img   = None
        self.refresh()
        self.status.config(text="Image loaded. Run Face Detection for Task 2.")

    def save_output(self):
        if self.output is None:
            messagebox.showwarning("Warning", "No output to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            cv2.imwrite(path, self.output)
            self.status.config(text="Saved.")

    def reset(self):
        if self.original is not None:
            self.output       = self.original.copy()
            self.before_faces = None
            self.before_time  = None
            self.before_img   = None
            self.refresh()
            self.status.config(text="Reset done.")

    # ---------------------------------------------------------------
    # APPLY FILTER TOOLS
    # ---------------------------------------------------------------

    def apply_tool(self):
        if self.original is None:
            messagebox.showwarning("Warning", "Upload an image first.")
            return

        name = self.tool.get()
        img  = self.output.copy()

        if name == "Brightness":
            v = simpledialog.askinteger("Brightness", "Value (−255 to 255):", initialvalue=50)
            if v is not None: self.output = brightness(img, v)

        elif name == "Contrast":
            a = simpledialog.askfloat("Contrast", "Alpha (0.1 – 3.0):", initialvalue=1.5)
            if a is not None: self.output = contrast(img, a)

        elif name == "Negative":
            self.output = negative(img)

        elif name == "Grayscale":
            self.output = grayscale(img)

        elif name == "Rotate 90°":
            self.output = rotate(img, 90)

        elif name == "Rotate Custom":
            a = simpledialog.askfloat("Rotate", "Angle (degrees):", initialvalue=45)
            if a is not None: self.output = rotate(img, a)

        elif name == "Zoom x2 (Nearest)":
            self.output = zoom(img, 2.0, cv2.INTER_NEAREST)

        elif name == "Zoom x2 (Bilinear)":
            self.output = zoom(img, 2.0, cv2.INTER_LINEAR)

        elif name == "Histogram Equalization":
            self.output = histogram_equalization(img)

        elif name == "Gamma Correction":
            g = simpledialog.askfloat("Gamma", "Gamma (0.1 – 5.0):", initialvalue=0.7)
            if g is not None: self.output = gamma_correction(img, g)

        elif name == "CLAHE":
            self.output = clahe(img)

        elif name == "Gaussian Blur":
            k = simpledialog.askinteger("Blur", "Kernel size (odd):", initialvalue=5)
            if k is not None: self.output = gaussian_blur(img, k)

        elif name == "Median Filter":
            k = simpledialog.askinteger("Median", "Kernel size (odd):", initialvalue=5)
            if k is not None: self.output = median_filter(img, k)

        elif name == "Sobel Edge":
            self.output = sobel_edge(img)

        elif name == "Canny Edge":
            t1 = simpledialog.askinteger("Canny", "Threshold 1:", initialvalue=100)
            t2 = simpledialog.askinteger("Canny", "Threshold 2:", initialvalue=200)
            if t1 and t2: self.output = canny_edge(img, t1, t2)

        self.refresh()
        self.status.config(text=f"Applied: {name}")

    # ---------------------------------------------------------------
    # FACE DETECTION - TASK 2
    # First click  → saves BEFORE score, shows it
    # Second click → runs AFTER score, shows full comparison table
    # ---------------------------------------------------------------

    def run_face_detection(self):
        if self.output is None:
            messagebox.showwarning("Warning", "Upload an image first.")
            return

        annotated, num_faces, elapsed = detect_faces(self.output)
        self.output = annotated
        self.refresh()

        # ---- FIRST RUN: save as BEFORE ----
        if self.before_faces is None:
            self.before_faces = num_faces
            self.before_time  = elapsed
            self.before_img   = annotated.copy()

            messagebox.showinfo(
                "Face Detection — BEFORE",
                f"BEFORE Result (bad image):\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"  Faces found  : {num_faces}\n"
                f"  Time         : {elapsed:.4f}s\n\n"
                f"Now apply your enhancement filters\n"
                f"(Brightness, CLAHE, Median Filter...)\n"
                f"then click Run Face Detection again."
            )
            self.status.config(text=f"BEFORE: {num_faces} face(s). Now enhance and run again.")

        # ---- SECOND RUN: show full comparison ----
        else:
            after_faces = num_faces
            after_time  = elapsed

            # Build comparison message
            if after_faces > self.before_faces:
                conclusion = "Enhancement IMPROVED detection! More faces found."
            elif after_faces == self.before_faces:
                conclusion = "Same number of faces found. Enhancement maintained accuracy."
            else:
                conclusion = "Fewer faces found after enhancement."

            msg = (
                f"{'':=<40}\n"
                f"       FACE DETECTION COMPARISON\n"
                f"{'':=<40}\n\n"
                f"                 BEFORE      AFTER\n"
                f"  Faces found  :  {self.before_faces:<10}  {after_faces}\n"
                f"  Time (s)     :  {self.before_time:.4f}      {after_time:.4f}\n\n"
                f"{'':─<40}\n"
                f"Conclusion:\n{conclusion}\n"
                f"{'':=<40}"
            )

            messagebox.showinfo("Face Detection — Comparison", msg)
            self.status.config(text=f"AFTER: {after_faces} face(s). Check the comparison!")

            # Reset so user can run a fresh experiment if they want
            self.before_faces = None
            self.before_time  = None
            self.before_img   = None

    # ---------------------------------------------------------------
    # DISPLAY HELPERS
    # ---------------------------------------------------------------

    def refresh(self):
        self._show(self.original, self.lbl_original)
        self._show(self.output,   self.lbl_output)
        self._histogram(self.original, self.ax_in,  self.cv_in)
        self._histogram(self.output,   self.ax_out, self.cv_out)

    def _show(self, img, label):
        rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        pil.thumbnail((480, 290))
        photo = ImageTk.PhotoImage(pil)
        label.config(image=photo, text="")
        label.image = photo

    def _histogram(self, img, ax, canvas):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ax.clear()
        ax.hist(gray.ravel(), bins=256, range=(0, 256), color="steelblue")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        canvas.draw()


# =============================================================================
# RUN
# =============================================================================

root = tk.Tk()
app  = VisionEditor(root)
root.mainloop()