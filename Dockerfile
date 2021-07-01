# step 1: xhost +local:docker
# docker run -u $(id -u):$(id -g) -it --rm -e DISPLAY=${DISPLAY} --gpus all -v $(pwd)/:/UNet:consistent -v /home/shailesh/Work/catkin_ws/src/simplepackage/src/IRLabeledDataset/:/IRLabeledDataset:consistent -v /tmp/.X11-unix:/tmp/.X11-unix -v ${HOME}/.Xauthority:/home/xterm/.Xauthority --hostname $(hostname) -p 0.0.0.0:6006:6006 tensorflow/tensorflow:2.4.0-gpu bash 
# docker run -u $(id -u):$(id -g) -it --rm -e DISPLAY=${DISPLAY} --gpus all -v $(pwd)/:/UNet:consistent -v /home/shailesh/Work/catkin_ws/src/simplepackage/src/IRLabeledDataset/:/IRLabeledDataset:consistent -v /tmp/.X11-unix:/tmp/.X11-unix -v ${HOME}/.Xauthority:/home/xterm/.Xauthority --hostname $(hostname) -p 0.0.0.0:6006:6006 unet:1 bash 

FROM tensorflow/tensorflow:2.4.0-gpu
ENV DISPLAY=$DISPLAY
ENV CUDA_VISIBLE_DEVICES=0

RUN set -eu pipefail && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get -y upgrade && \
    pip install Image && \
    apt-get -y install python-scipy python-matplotlib && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#docker build -t unet:1 .    