import tkinter as tk
from tkinter import ttk, Label, Button, font
from PIL import Image, ImageTk
import cv2
import depthai as dai
from mb_setup import setup_pipeline_mb, LABEL_MAP_MB
from yolo_setup import setup_pipeline_yolo, LABEL_MAP_YOLO

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OAK-D Depth Viewer")
        self.root.geometry("800x600")  # Set the window size to 800x600 pixels
        # self.root.configure(bg='black')  # Set the background color
        self.setup_widgets()

        self.pipeline = None
        self.device = None
        self.running = False

    def setup_widgets(self):
        self.model_var = tk.StringVar()
        self.color_var = tk.StringVar()

        # self.model_combobox = ttk.Combobox(self.root, textvariable=self.model_var, values=["yolov8", "mobile-ssd"])
        # self.model_combobox.pack(side="top", padx=30, pady=10)
        # self.model_combobox.current(0)

        # self.color_combobox = ttk.Combobox(self.root, textvariable=self.color_var, values=["Green", "Blue", "Red"])
        # self.color_combobox.pack(side="top", padx=20, pady=10)
        # self.color_combobox.current(0)
        # Create a font
        myFont = font.Font(family='Helvetica', size=14)  # Increase font size to increase combobox height

        style = ttk.Style(self.root)
        style.theme_use('clam')
        # Configure Combobox font
        # style.configure('TCombobox', font=myFont)

        # Configure a style for comboboxes
        style.configure('TCombobox',font=myFont, fieldbackground='light blue', background='blue', foreground='green', arrowcolor='black')

        self.model_combobox = ttk.Combobox(self.root, textvariable=self.model_var, values=["yolov8", "mobile-ssd"], style='TCombobox', height=10)
        self.model_combobox.pack(side="top", padx=20, pady=10)
        self.model_combobox.current(0)

        self.color_combobox = ttk.Combobox(self.root, textvariable=self.color_var, values=["Green", "Blue", "Red"], style='TCombobox', height=10)
        self.color_combobox.pack(side="top", padx=10, pady=10)
        self.color_combobox.current(0)

        self.start_btn = Button(self.root, text="Start", command=self.start_video)
        self.start_btn.pack(side="left", padx=10, pady=10)

        self.stop_btn = Button(self.root, text="Stop", command=self.stop_video)
        self.stop_btn.pack(side="right", padx=10, pady=10)

        self.label = Label(self.root)
        self.label.pack()

    def start_video(self):
        if not self.running:
            self.pipeline = self.setup_pipeline(self.model_var.get())
            if self.pipeline:
                self.device = dai.Device(self.pipeline)
                self.queues = {
                    'rgb': self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False),
                    'detections': self.device.getOutputQueue(name="detections", maxSize=4, blocking=False),
                    'depth': self.device.getOutputQueue(name="depth", maxSize=4, blocking=False),
                }
                self.running = True
                self.update_video()

    def stop_video(self):
        if self.running:
            self.running = False
            self.device.close()
            self.device = None

    def update_video(self):
        selected_color = self.color_var.get()
        if self.model_var.get() == 'yolov8':
            LABEL_MAP = LABEL_MAP_YOLO
        else:
            LABEL_MAP = LABEL_MAP_MB
        color_dict = {"Green": (0, 255, 0), "Red": (255, 0, 0), "Blue": (0, 0, 255)}
        color = color_dict[selected_color]
        if self.running:
            if all(q.has() for q in self.queues.values()):
                frames = {name: q.get() for name, q in self.queues.items()}

                frame = frames['rgb'].getCvFrame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = frames['detections'].detections

                # Process detections
                height, width, _ = frame.shape
                for detection in detections:
                    x1 = int(detection.xmin * width)
                    y1 = int(detection.ymin * height)
                    x2 = int(detection.xmax * width)
                    y2 = int(detection.ymax * height)

                    # Draw bounding box and add labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Bounding box
                    label_text = f"{LABEL_MAP[detection.label]}: {detection.confidence:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Annotations for spatial coordinates
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # frame = cv2.resize(frame, (640, 640)) 
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.config(image=imgtk)

            self.root.after(1, self.update_video)

    def setup_pipeline(self, model):
        if model == 'mobile-ssd':
            return setup_pipeline_mb()
        
        if model == 'yolov8':
            return setup_pipeline_yolo()

        return None  # Make sure to return a properly configured pipeline

def main():
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
