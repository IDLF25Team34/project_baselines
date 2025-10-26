FROM nvidia/cuda:11.6.1-devel-ubuntu20.04


# Set working directory
WORKDIR /workspace

# Prevent tzdata from prompting for input
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    libgl1 \
    libglib2.0-0

    # Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Accept Anaconda TOS for the default channels (non-interactive)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

CMD ["bash"]
