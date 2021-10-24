#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2, os
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        self.waypoints_2d = None 
        self.waypoint_tree = None

        self.frame = 0 # keeps count of image frames processed

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb) # use image_raw instead 

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.has_image = False 

        self.distance_to_next_light = None 
        self.car_wp_idx = None
        self.state_next_light = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(30) # can go as low as 30 (hertz) if desired
        while not rospy.is_shutdown():

            if (self.pose and self.waypoints != None and not self.has_image):

                light_wp, state = self.process_traffic_lights()
                # rospy.logwarn("Closest light wp: {0} \n And light state: {1}".format(light_wp, state))
                '''
                Publish upcoming red lights at camera frequency.
                Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
                of times till we start using it. Otherwise the previous stable state is
                used.
                '''
                if self.state != state: # state of light based on processing
                    self.state_count = 0
                    self.state = state
                elif self.state_count >= STATE_COUNT_THRESHOLD:
                    self.last_state = self.state
                    light_wp = light_wp if state == TrafficLight.RED else -1
                    self.last_wp = light_wp
                    self.upcoming_red_light_pub.publish(Int32(light_wp))
                else:
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                self.state_count += 1

            # toggle so that above code runs if camera turned off (use provided traffic light info)
            if self.has_image:
                self.has_image = False

            rate.sleep()    

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # if not self.waypoints_2d:
        self.waypoints = waypoints
        self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] 
                                for waypoint in waypoints.waypoints]
        self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):

        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # if self.distance_to_next_light <= 300:
        #     path_name = '../training_images'  
        #     if not os.path.exists(path_name):
        #         os.mkdir(path_name)
        #     img_name = "img{0:5d}{1}.jpg".format(self.car_wp_idx, self.state)            
        #     writeStatus = cv2.imwrite(os.path.join(path_name, img_name), cv_image)
        #     # rospy.logwarn("image name: {0} write status: {1}".format(img_name, writeStatus))

          
        # self.frame += 1
        # if self.frame % 3 != 0:
        #     return 
        return 



    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            return light.state # until classification built 
            # self.prev_light_loc = None
            # return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None 
        line_wp_idx = None 
        # light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)

            for i, light in enumerate(self.lights):
                # Get stop line waypoint index 
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx 
                if d >= 0 and d < diff:
                    diff = d 
                    closest_light = light 
                    line_wp_idx = temp_wp_idx
        
        if closest_light:
            state = self.get_light_state(closest_light)

            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            self.distance_to_next_light = line_wp_idx - car_wp_idx
            self.car_wp_idx = car_wp_idx
            self.state_next_light = state

            # rospy.logwarn("tl closest index: {0} line wp index: {1} difference: {2} state: {3}".format(car_wp_idx, line_wp_idx, car_wp_idx-line_wp_idx, state))

            return line_wp_idx, state

        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
