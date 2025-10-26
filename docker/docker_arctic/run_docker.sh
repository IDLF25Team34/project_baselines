# xhost +local:root

# docker run -dit \
#     --gpus all \
#     --env="DISPLAY=$DISPLAY" \
#     --env="QT_X11_NO_MITSHM=1" \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     -v "$(pwd)":/workspace/idl_project \
#     -w /workspace/idl_project \
#     --name arctic-train \
#     arctic-train:v3

xhost +local:root
docker run -dit \
  --gpus all \
  --net=host \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --env="XDG_RUNTIME_DIR=/tmp/runtime-root" \
  --env="NVIDIA_VISIBLE_DEVICES=all" \
  --env="NVIDIA_DRIVER_CAPABILITIES=all,display,graphics,utility,compute" \
  -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v "$(pwd)":/workspace/idl_project \
  -w /workspace/idl_project \
  --name arctic-train \
  arctic-train:v4