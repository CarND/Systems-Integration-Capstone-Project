#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import TrafficLightArray
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32 as Float
from std_msgs.msg import Int32

from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import numpy as np 

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''
"""
source devel environment develsetup.sh and echo final waypoints rostopic echo /final_waypoints to make sure node is publishing final waypoints
"""

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
# initialize waypoints_2d
# waypoints_2d = None 

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        self.pose_error_pub = rospy.Publisher('/current_pose/error', Float, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None # PoseStamped msg type
        self.base_waypoints = None # Lane msg type
        self.waypoints_2d = None 
        self.waypoint_tree = None

        self.stopline_wp_idx = -1 # waypoints used while no stopline_wp_idx  

        # Be sure to limit the top speed of the vehicle to the km/h velocity set by the velocity rosparam in waypoint_loader. Reviewers will test on the simulator with an adjusted top speed.
        self.MAX_DECEL = 5 # acceleration should not exceed 10 m/s^2 and jerk should not exceed 10 m/s^3.

        self.loop()
        # rospy.spin()

    def loop(self):
        rate = rospy.Rate(30) # can go as low as 30 (hertz) if desired
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                # self.publish_waypoints(closest_waypoint_idx)
                self.publish_waypoints()

                # calculate error for steer (yaw) correction
                offset = 50
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                wp_y = self.waypoints_2d[closest_waypoint_idx][1]
                y = self.pose.pose.position.y 
                lane_err = y - wp_y - offset
                self.pose_error_pub.publish(lane_err)
            rate.sleep()    

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x 
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1] # first 1 gives closes pt, second 1 returns index of [position, index]

        # check if closest is ahead or behind vehicle 
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        # lane.header not used, can ignore
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS 
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            # rospy.logwarn("regular wp closest index: {0} line wp index: {1} difference: {2}".format(closest_idx, self.stopline_wp_idx, closest_idx-self.stopline_wp_idx))            
            lane.waypoints = base_waypoints
        else:
            # rospy.logwarn("decelerate wp closest index: {0} line wp index: {1} difference: {2}".format(closest_idx, self.stopline_wp_idx, closest_idx-self.stopline_wp_idx))            
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
            # rospy.logwarn("decelerate waypoints: {0}".format([wp.twist.twist.linear.x for wp in lane.waypoints]))


        return lane
    
    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints): 

            p = Waypoint()
            p.pose = wp.pose 
            stop_idx = max(min(self.stopline_wp_idx - closest_idx - 8, LOOKAHEAD_WPS), 0) # Two waypoints back from line so front of car stops at lin 
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * self.MAX_DECEL * dist) # consider using S curve or constant
            if vel < 1.:
                vel = 0.
            # coordinates for linear velocity are vehicle-centered, so only the x-direction linear velocity should be nonzero.
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints 
        if not self.waypoints_2d:
            # fix for list comprehension
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] 
                                    for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement   
        self.stopline_wp_idx = msg.data 
        # rospy.logwarn("Traffic_cb stopline_wp_inx: {0} \n".format(msg.data))

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. No need to implement for current project.
        pass    

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
