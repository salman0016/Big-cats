
# Big Cats detection on Yolov5 using Jetson Nano 2gb 

# Aim And Objectives :-

Aim :-

To create a Big Cats detection system which will detect cats face and mark. Then it will classify which species it belongs.

Objective :-

• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• To show on the optical viewfinder of the camera module whether a cat belongs to which species.

# Abstract :-
• A cat face is classified whether a cat is belonging to which species and is detected by the live feed from the system’s camera.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

• Because of human population growth and activity such as habitat removal and climate change, all big cat species populations are on decline. Conservation of big cats is needed to health of ecosystems as these are apex predators. 

• The purpose of this project is to observe big cats behaviour.

# Introduction :-
• This project is based on a Big Cats detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.

• This project can also be used to gather information about which species does the cats belong.

• Big cats can be classified into Tiger, Jaguar and Cheetah based on the image annotation we give in roboflow. 

• Cats detection sometimes become difficult as face or body are covered up by grass,tress and rock thereby making big cats detection difficult. However, training in Roboflow has allowed us to crop images and change the contrast of certain images to match the time of day for better recognition by the model.

• Neural networks and machine learning have been used for these tasks and have obtained good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Big Cats detection as well.


# Jetson Nano Compatibility :-
• The power of modern AI is now available for makers, learners, and embedded developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

# Jetson Nano 2gb :-


# Proposed System :-

1] Study basics of machine learning and image recognition.

2]Start with implementation

• Front-end development

• Back-end development

3] Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether the cat belongs to which spices.

4] use datasets to interpret the object and suggest whether the cat on the camera’s viewfinder belongs to which spices.

# Methodology :-
The Big cats detection system is a program that focuses on implementing real time Cat detection. It is a prototype of a new product that comprises of the main module: Cats detection and then showing on viewfinder whether the cat belongs to which species. Big Cats Detection Module

This Module is divided into two parts:

1] Cheetah detection :-

• Ability to detect the location of a cat's face in any input image or frame. The output is the bounding box coordinates on the detected face of a cat.

• For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.

• This Datasets identifies cat’s face in a Bitmap graphic object and returns the bounding box image with annotation of Cheetah, Jaguar or Tiger present in each image.

2] jaguar Detection :-

• Recognition of the face and whether the Cat belongs to which species.

• Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward. 

3] Tiger Detection :-

• Recognition of face or mark whether the cat belongs to which species.

• YOLOv5 was used to train and test our model for whether the cat belongs to which species. We trained it for 149 epochs and achieved an accuracy of approximately 92%.

# Installation :-

Initial Configuration :-

sudo apt-get remove --purge libreoffice*

sudo apt-get remove --purge thunderbird*

Create Swap :-

udo fallocate -l 10.0G /swapfile1


sudo chmod 600 /swapfile1

sudo mkswap /swapfile1

sudo vim /etc/fstab

#make entry in fstab file

/swapfile1	swap	swap	defaults	0 0

Cuda env in bashrc :-

vim ~/.bashrc

#add this lines

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

Update & Upgrade :-

sudo apt-get update

sudo apt-get upgrade

Install some required Packages :-

sudo apt install curl

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

sudo python3 get-pip.py

sudo apt-get install libopenblas-base libopenmpi-dev

sudo pip3 install pillow

Install Torch :-

curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl

mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 

sudo python3 -c "import torch; print(torch.cuda.is_available())"

Install Torchvision :-

git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision

cd torchvision/

sudo python3 setup.py install

Clone Yolov5 :-

git clone https://github.com/ultralytics/yolov5.git

cd yolov5/

sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1

sudo pip3 install -r requirements.txt

Download weights and Test Yolov5 Installation on USB webcam :-

sudo python3 detect.py

sudo python3 detect.py --weights yolov5s.pt  --source 0

# Big Cats Dataset Training :-

# We used Google Colab And Roboflow :-
train your model on colab and download the weights and past them into yolov5 folder link of project colab file given in repo

# Running Cats Detection Model :-

source '0' for webcam

!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0

# Demo :-


# Advantages :-
• Big cats detection system will be of great help in locating cats in jungle safari.

• Helmet detection system shows whether the cat in viewfinder of camera module is Cheetah, Jaguar or Tiger with good accuracy.

• It can then convey it to authorities like forest officer or the data about the respective cat where he is relocatting and it can help the forest department to spot the big cat easily.

• When completely automated no user input is required and therefore works with absolute efficiency and speed.

• It can work around the clock and therefore becomes more cost efficient.

# Application :-
• Detects a cat’s face and then checks whether the cat belongs to which species in each image frame or viewfinder using a camera module.

• Can be used anywhere in forest as the cat usually roam and Big cats detection becomes even more accurate.

• Can be used as a reference for other ai models based on Big cats Detection.

# Future Scope :-
• As we know technology is marching towards automation, so this project is one of the step towards automation.

• Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.

• Big cats detection will become a necessity in the future due to decrease in population and hence our model will be of great help to locate the cats in an efficient way.

# Conclusion :-
• In this project our model is trying to detect a cat’s face or body and then showing it on viewfinder, live as whether cat belongs to which species as we have specified in Roboflow.

• The model tries to solve the problem of severe injuries and attack of cats to human that occure in forest  and thus protects a person’s life.

• The model is efficient and highly accurate and hence reduces the workforce required.

# Reference :-
1] Roboflow:- https://roboflow.com/

2] Datasets or images used :- https://www.gettyimages.ae/photos/big-cats?assettype=image&phrase=big%20cats&sort=mostpopular&license=rf%2Crm

3] Google images

# Articles :-
1 :- https://www.thestatesman.com/environment/it-is-extremely-important-1503094228.html

2 :- https://wwf.ca/stories/six-reasons-why-the-world-needs-to-protect-big-cats/
