import numpy as np
import yaml
from typing import Tuple
import argparse

# ROS2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy

from topic_names import WAYPOINT_TOPIC, REACHED_GOAL_TOPIC, VEL_TOPIC
from ros_data import ROSData
from utils import clip_angle 

WAYPOINT_TIMEOUT = 1  # seconds


# GLOBALS
vel_msg = Twist()
waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
reached_goal = False
reverse_mode = False
current_yaw = None

def clip_angle(theta) -> float:
    """Clip angle to [-pi, pi]"""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi

def pd_controller(waypoint: np.ndarray, max_v, max_w, dt, eps=1e-8) -> Tuple[float]:
    """PD controller for the robot"""

    # if waypoint[0] != 0.0:
    #     pdb.set_trace()
    assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint
    # this controller only uses the predicted heading if dx and dy are near zero
    if len(waypoint) == 4 and np.abs(dx) < eps and np.abs(dy) < eps:
        v = 0
        w = clip_angle(np.arctan2(hy, hx)) / dt
    elif np.abs(dx) < eps:
        v = 0
        w = np.sign(dy) * np.pi / (2 * dt)
    else:
        v = dx / dt
        w = np.arctan(dy / dx) / dt
    v = np.clip(v, 0, max_v)
    w = np.clip(w, -max_w, max_w)
    return v, w

class PDControllerNode(Node):
    def __init__(self, args):
        super().__init__("pd_controller")
        self.vel_msg = Twist()
        self.waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
        self.reached_goal = False
        self.reverse_mode = False
        self.args = args
        # CONSTS
        config_path = "./deployment/config/robot.yaml"
        with open(config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        self.rate = robot_config["frame_rate"]

        robot_config = robot_config[args.robot]
        self.max_v = robot_config["max_v"]
        self.max_w = robot_config["max_w"]

        self.dt = 1 / self.rate
        EPS = 1e-8
        FLIP_ANG_VEL = np.pi / 4

        self.waypoint_sub = self.create_subscription(Float32MultiArray, 
                                                     WAYPOINT_TOPIC, 
                                                     self.callback_drive, 
                                                     qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                                                              history=QoSHistoryPolicy.KEEP_LAST,
                                                                              depth=10))
        self.reached_goal_sub = self.create_subscription(Bool, 
                                                         REACHED_GOAL_TOPIC, 
                                                         self.callback_reached_goal, 
                                                         10)
        self.vel_out = self.create_publisher(Twist, 
                                             VEL_TOPIC,
                                             10)

        self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)
        self.get_logger().info("Registered with master node. Waiting for waypoints...")

    def callback_drive(self, waypoint_msg: Float32MultiArray):
        """Callback function for the waypoint subscriber"""
        self.get_logger().info("Setting waypoint")
        self.waypoint.set(waypoint_msg.data)

    def callback_reached_goal(self, reached_goal_msg: Bool):
        """Callback function for the reached goal subscriber"""
        self.reached_goal = reached_goal_msg.data

    def timer_callback(self):
        if self.reached_goal:
            self.vel_out.publish(self.vel_msg)
            self.get_logger().info("Reached goal! Stopping...")
            rclpy.shutdown()
            return
        
        if self.waypoint.is_valid(verbose=True):
            v, w = pd_controller(self.waypoint.get(), self.max_v, self.max_w, self.dt, eps=1e-8)
            if self.reverse_mode:
                v *= -1
            self.vel_msg.linear.x = v
            self.vel_msg.angular.z = w
            self.get_logger().info(f"Publishing new velocity: {v}, {w}")
        self.vel_out.publish(self.vel_msg)

def main():
    parser = argparse.ArgumentParser(description="Run the Path Manager")
    parser.add_argument("-r", "--robot", type=str, help="Robot Name",
                        default="ghost")
    args = parser.parse_args()
    print("robot name: ", args.robot)
    rclpy.init()
    pd_controller_node = PDControllerNode(args)
    try:
        rclpy.spin(pd_controller_node)
    except KeyboardInterrupt:
        pass
    finally:
        pd_controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()