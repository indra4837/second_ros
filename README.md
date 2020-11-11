# Second 

This package contains the second model that performs 3D object detection from pointcloud data.

![Video_Result2](docs/results.gif)

---
## Setting up the environment and installing dependencies

### Current Environment

- ROS Melodic
- Ubuntu 18.04
- NVIDIA GeForce 940MX

Please install all dependencies required for the [second.pytorch model.](https://github.com/traveller59/second.pytorch)

Please note that this repo requires Python 3.6+. If you are using ROS Melodic, please build ROS from source to ensure rospy runs with Python3.

## Setting up the package

### Clone project into catkin_ws and build it

``` 
$ cd ~/catkin_ws && catkin_make
$ source devel/setup.bash
```

### Download rosbag files for testing

To download rosbags for testing, please follow the [link](https://github.com/tomas789/kitti2bag) to get a kitti.bag for testing.

## Using the package

### Running the package

```
$ roslaunch second_ros second_kitti.launch
```

## Future improvements
- [ ] Test model inference on NVIDIA Jetson Xavier
- [ ] Inference with ONNX or TensorRT
- [ ] Add pointpillar/TANET support

## Licenses and References
I referenced original implementation of [second.pytorch](https://github.com/traveller59/second.pytorch) which is under MIT License
