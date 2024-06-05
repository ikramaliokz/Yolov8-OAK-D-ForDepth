FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set the DEBIAN_FRONTEND to noninteractive to avoid user interaction during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your application code
COPY yolov8_depth_using_OAKD.py /app/yolov8_depth_using_OAKD.py
WORKDIR /app

# Command to run your application
CMD ["python3", "yolov8_depth_using_OAKD.py"]
