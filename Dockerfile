# Use the official Python image as the base
FROM python:3.9

# Set environment variables for writable cache directories
ENV MPLCONFIGDIR=/tmp/matplotlib_cache
ENV XDG_CACHE_HOME=/tmp/.cache

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    stable-baselines3==2.3.2 \
    numpy==2.0.0 \
    gymnasium==0.29.1 \
    gym==0.26.2 \
    opencv-python-headless \
    Pillow \
    imageio \
    typing-extensions \
    fastapi \
    uvicorn


    # Uninstall fontconfig to prevent cache warnings
RUN apt-get remove -y fontconfig && \
apt-get autoremove -y && \
rm -rf /var/lib/apt/lists/*

# Expose the port your application will run on
EXPOSE 7860

# Define the command to run your application
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860"]
