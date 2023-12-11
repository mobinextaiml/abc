import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, Button, filedialog
from shapely.geometry import Polygon
from ultralytics import YOLO
from tracker import Tracker

class VideoProcessor:
    def __init__(self):
        self.polygon_points = []
        self.video_path = ""
        self.clone = None

    def draw_polygon(self, image, points):
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], True, (0, 255, 0), 2)

    def store_polygon_coordinates(self, points, filename):
        with open(filename, "w") as file:
            for point in points:
                file.write(f"{point[0]},{point[1]}\n")

    def load_polygon_coordinates(self, filename):
        points = []
        with open(filename, "r") as file:
            for line in file:
                x, y = map(int, line.strip().split(","))
                points.append((x, y))
        return points

    def upload_video(self):
        self.video_path = filedialog.askopenfilename()
        cap = cv2.VideoCapture(self.video_path)
        ret, first_frame = cap.read()  # Add this line to capture the first frame of the video

        # Load the YOLO model
        yolo_model = YOLO('best (18).pt')

        
        # Define the class list
        my_file = open("coco.txt", "r")
        data = my_file.read()
        class_list = data.split("\n")

        # Define the tracker
        tracker = Tracker()

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.polygon_points.append((x, y))
                if len(self.polygon_points) > 1:
                    self.draw_polygon(self.clone, self.polygon_points)
            elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
                self.polygon_points.append((x, y))
                self.draw_polygon(self.clone, self.polygon_points)
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.polygon_points) > 2:
                    self.draw_polygon(self.clone, self.polygon_points)
                    cv2.line(self.clone, self.polygon_points[-1], self.polygon_points[0], (0, 0, 255), 2)  # Close the polygon
                self.store_polygon_coordinates(self.polygon_points, "polygon_coordinates.txt")

        self.clone = first_frame.copy()
        cv2.namedWindow("Draw Polygon", cv2.WINDOW_NORMAL)  # Set the window to be resizable
        cv2.resizeWindow("Draw Polygon", 800, 800)  # Set the window size for better visibility
        cv2.setMouseCallback("Draw Polygon", on_mouse)

        while True:
            cv2.imshow("Draw Polygon", self.clone)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Escape key
                break

        cv2.destroyAllWindows()

        screen_res = (1920, 1080)
        scale_width = screen_res[0] / first_frame.shape[1]
        scale_height = screen_res[1] / first_frame.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(first_frame.shape[1] * scale)
        window_height = int(first_frame.shape[0] * scale)

        cv2.namedWindow("Person detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Person detection", window_width, window_height)

        while True:
            ret, frame = cap.read()
            frame_copy = frame.copy()
            if not ret:
                break
            frame = cv2.resize(frame, (window_width, window_height))
            results = yolo_model.predict(frame)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")
            list = []
            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                c = class_list[d]
                if 'person' in c:
                    list.append([x1, y1, x2, y2])
            bbox_idx = tracker.update(list)
            for bbox in bbox_idx:
                x3, y3, x4, y4, id = bbox
                a = (x3 + x4)/2
                b = (y3 + y4)/2
                result = cv2.pointPolygonTest(np.array(self.polygon_points, np.int32), ((a, b)), False)
                if result <= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                    cv2.putText(frame, 'person', (x3, y3-15), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
            self.draw_polygon(frame, self.polygon_points)
            cv2.imshow("Person detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

root = Tk()
root.geometry("400x200")

processor = VideoProcessor()

upload_button = Button(root, text="Upload Video", command=processor.upload_video)
upload_button.pack()

root.mainloop()
