#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from pyquaternion import Quaternion
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

import sys
import time
sys.path.append("/home/indra/rospy3_melodic/src/second_ros/second.pytorch")

import numpy as np
import math
import pickle
from pathlib import Path

import torch
from google.protobuf import text_format
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool

class Second_ROS:
    def __init__(self):
        config_path, ckpt_path = self.init_ros()
        self.init_second(config_path, ckpt_path)

    def init_second(self, config_path, ckpt_path):
        """ Initialize second model """
        
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        input_cfg = config.eval_input_reader
        model_cfg = config.model.second
        #config_tool.change_detection_range_v2(model_cfg, [-50, -50, 50, 50])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")

        self.net = build_network(model_cfg).to(self.device).eval()
        self.net.load_state_dict(torch.load(ckpt_path))
        target_assigner = self.net.target_assigner
        self.voxel_generator = self.net.voxel_generator

        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]

        self.anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
        self.anchors = torch.tensor(self.anchors, dtype=torch.float32, device=self.device)
        self.anchors = self.anchors.view(1, -1, 7)


    def init_ros(self):
        """ Initialize ros parameters """

        self.sub_velo = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)
        self.pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=1)

        config_path = rospy.get_param("/config_path", "/home/indra/fyp_perception/traveller/second.pytorch/second/configs/car.fhd.config")
        ckpt_path = rospy.get_param("/ckpt_path", "/home/indra/Documents/pretrained_models_v1.5/car_onestage/voxelnet-27855.tckpt")
        
        #config_path = rospy.get_param("/config_path", "/home/indra/rospy3_melodic/src/second_ros/config/all.fhd.config")
        #ckpt_path = rospy.get_param("/ckpt_path", "/home/indra/rospy3_melodic/src/second_ros/trained_models/voxelnet-99040.tckpt")

        return config_path, ckpt_path

    def inference(self, points):
        num_features = 4
        points = points.reshape([-1, num_features])
        print(points.shape)
        dic = self.voxel_generator.generate(points)
        voxels, coords, num_points = dic['voxels'], dic['coordinates'], dic['num_points_per_voxel']
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        
        input_points = {
        "anchors": self.anchors,
        "voxels": voxels,
        "num_points": num_points,
        "coordinates": coords,
        }

        pred = self.net(input_points)[0]
        boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        label = pred["label_preds"].detach().cpu().numpy()

        return boxes_lidar, scores, label



    def lidar_callback(self, msg):
        """ Captures pointcloud data and feed into second model for inference """

        #arr_box = BoundingBoxArray()

        pcl_msg = pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z","intensity","ring"))
        np_p = np.array(list(pcl_msg), dtype=np.float32)
        #print("np_p shape: ", np_p.shape)
        #np_p = np.delete(np_p, -1, 1)  #  delete "ring" field
        
        # convert to xyzi point cloud
        x = np_p[:, 0].reshape(-1)
        y = np_p[:, 1].reshape(-1)
        z = np_p[:, 2].reshape(-1)
        if np_p.shape[1] == 4: # if intensity field exists
            i = np_p[:, 3].reshape(-1)
        else:
            i = np.zeros((np_p.shape[0], 1)).reshape(-1)
        cloud = np.stack((x, y, z, i)).T
        
        # start processing
        tic = time.time()
        boxes_lidar, scores, label = self.inference(cloud)
        toc = time.time()
        fps = 1/(toc-tic)
        rospy.loginfo("FPS: %f", fps)

        num_detections = len(boxes_lidar)
        arr_bbox = BoundingBoxArray()

        for i in range(num_detections):
            bbox = BoundingBox()

            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()

            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2]) + float(boxes_lidar[i][5]) / 2
            bbox.dimensions.x = float(boxes_lidar[i][3])  # width
            bbox.dimensions.y = float(boxes_lidar[i][4])  # length
            bbox.dimensions.z = float(boxes_lidar[i][5])  # height

            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = label[i]

            arr_bbox.boxes.append(bbox)
        
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = rospy.Time.now()
        #print("Number of detections: {}".format(num_detections))
        
        self.pub_bbox.publish(arr_bbox)


if __name__ == '__main__':
    sec = Second_ROS()
    rospy.init_node('second_ros_node', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")
