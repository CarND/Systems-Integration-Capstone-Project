
import rospy 
from yaw_controller import YawController 
from pid import PID 
from lowpass import LowPassFilter 

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, 
        accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        kp = -0.6 #-0.4 #-0.2
        ki = 0.1 
        kd = -0.75 

        mn = max_lat_accel 
        mx = max_steer_angle 

        self.yaw_corrector = PID(kp, ki, kd, mn, mx) 
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        kp = 0.3
        ki = 0.1 
        kd = 0. 
        mn = 0.01 
        mx = 0.6 
        self.throttle_controller = PID(kp, ki, kd, mn, mx) # values determined experimentally

        tau = 0.5 # 1/(2pi*tau) = cutoff frequency 
        ts = .02 # Sample time 
        # lowpassfilter filters out high frequency noise from velocity values
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass 
        self.fuel_capacity = fuel_capacity 
        self.brake_deadband = brake_deadband 
        self.decel_limit = decel_limit # given - set for comfort
        self.accel_limit = accel_limit # given - set for comfort
        self.wheel_radius = wheel_radius 

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel, pose_error):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        # 0.1 is lowest speed of car m/s, max lateral accel, max steer angle passed in dbw.py - need to forward them to yaw controller 
        if not dbw_enabled:
            self.throttle_controller.reset()
            self.yaw_corrector.reset()
            return 0., 0., 0.
        current_vel = self.vel_lpf.filt(current_vel)
        
        # rospy.logwarn("Angular vel: {0}".format(angular_vel))
        # rospy.logwarn("Target velocity: {0}".format(linear_vel))
        # rospy.logwarn("Target angular velocity: {0}".format(angular_vel))
        # rospy.logwarn("Current velocity: {0}".format(current_vel))
        # rospy.logwarn("Filtered velocity: {0}".format(self.vel_lpf.get()))        

        correction = self.yaw_corrector.run(pose_error) 
        self.yaw_controller.update_steer_ratio(correction)
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_error = linear_vel - current_vel 
        self.last_vel = current_vel 

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time 
        self.last_time = current_time

        throttle = abs(self.throttle_controller.step(vel_error, sample_time))
    
        brake = 0

        # rospy.logwarn("current vel: {0} linear vel: {1} vel error: {2}".format(current_vel, linear_vel, vel_error))            

        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 700 #N*m - to hold car in place if stopped at light. Accel - 1 m/s^2

        elif vel_error < 0:
            throttle = 0 
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # Torque N*m            
            # rospy.logwarn("decel: {0} brake: {1} decel limit: {2}".format(decel, brake, self.decel_limit))            

        return throttle, brake, steering
