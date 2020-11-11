#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sys
sys.path.append("/home/indra/traveller/second.pytorch")
import torch
from google.protobuf import text_format
from second.utils import simplevis
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool


if __name__ == '__main__':
    #config_path = "/home/indra/traveller/second.pytorch/second/configs/car.fhd.config"
    config_path = "/home/indra/traveller/second.pytorch/second/configs/all.fhd.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    # config_tool.change_detection_range(model_cfg, [-50, -50, 50, 50])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #ckpt_path = "/home/indra/Documents/pretrained_models_v1.5/car_onestage/voxelnet-27855.tckpt"
    ckpt_path = "/home/indra/Documents/second_models/voxelnet-99040.tckpt"
    net = build_network(model_cfg).to(device).eval()
    net.load_state_dict(torch.load(ckpt_path))
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator


    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
    feature_map_size = [*feature_map_size, 1][::-1]

    #Here is the thing I changed
    v_path = '/home/indra/Documents/Kitti/object/training/velodyne_reduced/000001.bin'
    anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = anchors.view(1, -1, 7)

    points = np.fromfile(
        v_path, dtype=np.float32, count=-1)
    #print(points.shape)
    points = np.fromfile(
        v_path, dtype=np.float32, count=-1).reshape([-1, 5])
    points = points[:, :4]
    #print(points.shape)
    dic = voxel_generator.generate(points)
    voxels, coords, num_points = dic['voxels'], dic['coordinates'], dic['num_points_per_voxel']

    #print(voxels.shape)
    # add batch idx to coords
    coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.tensor(num_points, dtype=torch.int32, device=device)


    example = {
        "anchors": anchors,
        "voxels": voxels,
        "num_points": num_points,
        "coordinates": coords,
    }

    net.eval()

    #second_trt = torch2trt(net, example)
    #torch.save(second_trt.state_dict(), "/home/indra/second.trt")

    torch.onnx.export(net, [anchors, voxels, num_points, coords], "/home/indra/second.onnx", verbose=False)
    #print('second.onnx transfer is successful')
    
    #pred = net(example)[0]
    #print(pred)

    #boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
    #vis_voxel_size = [0.1, 0.1, 0.1]
    #vis_point_range = [-50, -30, -3, 50, 30, 1]
    #bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    #bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)

    #plt.imshow(bev_map)
    #plt.savefig('map.png')