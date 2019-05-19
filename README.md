# 3D Perception - Robotics

In this project, I implemented a 3D perception pipeline using ROS, PCL to identify target objects from a given “Pick-List” in a particular order and successfully complete a tabletop pick and place operation using PR2 robot.

The PR2 used in the project is outfitted with an RGB-D sensor.

# Project Setup
For this setup, catkin_ws is the name of active ROS Workspace. To create one:

```sh
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
```

Now that you have a workspace, clone or download this repo into the src directory of your workspace:
```sh
$ cd ~/catkin_ws/src
$ git clone https://github.com/umerjamil16/3D-Perception-Project.git
```

Now install missing dependencies using rosdep install:
```sh
$ cd ~/catkin_ws
$ rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
```
Build the project:
```sh
$ cd ~/catkin_ws
$ catkin_make
```
Also add following to the .bashrc file
```
export GAZEBO_MODEL_PATH=~/catkin_ws/src/Perception-Project/pr2_robot/models:$GAZEBO_MODEL_PATH
```

Following line can be added to your .bashrc to auto-source all new terminals
```
source ~/catkin_ws/devel/setup.bash
```

To run the demo:
```sh
$ cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts
$ chmod u+x pr2_safe_spawner.sh
$ ./pr2_safe_spawner.sh
```
![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)



In the RViz window, you should see the robot and a partial collision map displayed:

![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Proceed through the demo by pressing the ‘Next’ button on the RViz window when a prompt appears in  active terminal

The demo ends when the robot has successfully picked and placed all objects into respective dropboxes (though sometimes the robot gets excited and throws objects across the room!)

Close all active terminal windows using **ctrl+c** before restarting the demo.

You can launch the project scenario through:
```sh
$ roslaunch pr2_robot pick_place_project.launch
$ rosrun pr2_robot object-rec.py
```

Note: The robot is a bit moody at times and might leave objects on the table or fling them across the room :D
