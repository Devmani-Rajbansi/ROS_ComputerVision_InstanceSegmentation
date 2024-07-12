#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import sort


class Object_Detection():
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.model = YOLO("/home/touhid/catkin_ws/src/ridgeback/ridgeback/ridgeback_navigation/src/helper_func/OB_SAM_Small_.pt")
        rospy.Subscriber('/webcam', Image, self.image_callback)
        self.results = []

        self.tracker = sort.Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.total_count = []
        self.limits = [320, 1, 320, 479]

    def image_callback(self, msg):
        print("Image callback executed")
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.results = self.model(frame, save=False, conf=0.6)

        helper_result = self.results[0]
        self.playground()

        if True:
                self.plot_result()

    def playground(self):
        print("Playground called")
        detections = np.empty((0, 4))
        x1, x2, y1, y2 = 0, 0, 0, 0
        for result in self.results:
            print("Object detected")
            boxes=result.boxes.cpu().numpy()
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                cls = box.cls
                class_names = ['Bin_red','Bin_yellow']
                output_index = int(cls[0])
                class_name = class_names[output_index]
                if class_name == "Bin_red" or class_name == "Bin_yellow":
                    currentArray = np.array([x1, y1, x2, y2])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = self.tracker.update(detections)  
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            print(cx, cy)
            if self.limits[0] - 15 < cx < self.limits[2] + 15 and self.limits[1] < cy < self.limits[3]:
                if self.total_count.count(id) == 0:
                    self.total_count.append(id)
                    rospy.set_param("/bath/movement_pause", 1)
                    rospy.sleep(10)
                    rospy.set_param("/bath/movement_pause", 0)

        print("count: ", len(self.total_count))



    def plot_result(self):
            annotated_frame = self.results[0].plot()
            cv2.imshow("Annotated Frame", annotated_frame)
            cv2.waitKey(1)

    def run(self):
        pass

if __name__ == '__main__':
    rospy.init_node('image_subscriber')
    obdt = Object_Detection()
    rospy.spin()


