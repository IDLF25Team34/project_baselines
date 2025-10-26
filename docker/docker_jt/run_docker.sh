xhost +local:root

docker run -dit \
    --gpus all \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$(pwd)":/workspace/idl_project \
    -v "/home/student/Documents/parth/idl_project/arctic":/workspace/arctic \
    -w /workspace/idl_project \
    --name jointtransformer \
    jointtransformer
