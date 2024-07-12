#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

bridge = CvBridge()

def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Webcam Grayscale Image", gray_image)
        cv2.waitKey(1)
    except CvBridgeError as e:
        print(e)

def listener():
    rospy.init_node('image_subscriber', anonymous=True)
    rospy.Subscriber('/webcam', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

