FROM python:3.10-slim  
# minimal version, no unnecessary packages

WORKDIR /workspace
# Creates a folder /workspace inside the container

ENV PYTHONDONTWRITEBYTECODE=1 \  
# """stops Python from creating .pyc cache files (keeps container clean)"""
    PYTHONUNBUFFERED=1
# prints logs instantly to terminal instead of buffering them


RUN apt-get update && apt-get install -y --no-install-recommends \          
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch (~200MB instead of 755MB)
RUN pip install --no-cache-dir --timeout=1000 --retries=10 \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages from PyPI
RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    numpy==1.26.4 \
    Pillow==10.3.0 \
    scikit-learn==1.4.2 \
    matplotlib==3.8.4 \
    jupyter==1.0.0 \
    notebook==7.1.3 \
    opencv-python-headless==4.9.0.80

COPY mask_detection_pytorch.ipynb .

EXPOSE 8888

CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]