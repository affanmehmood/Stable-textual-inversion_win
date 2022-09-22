# FROM nvidia/cuda:10.2-base
# FROM python:3.7.9

FROM nvidia/cuda:10.2-cudnn7-devel
# FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04
# FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# ARG IMG_TAG=1.8.1-cuda10.2-cudnn7-devel
# ARG IMG_REPO=pytorch
#
# FROM pytorch/$IMG_REPO:$IMG_TAG

# RUN apt-get -y update && apt-get -y install git gcc llvm-9-dev cmake libaio-dev vim wget

# CMD nvidia-smi
# set up environment
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3.7
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install make
RUN apt-get update
RUN apt-get install zlib1g-dev -y
RUN apt-get install libjpeg-dev -y
RUN apt-get install git -y

# RUN apt-get install -y nvidia-docker2
# RUN apt-get -y install nvcc


WORKDIR /
# COPY ./requirements.txt ./

# ENV CUDA_HOME="/usr/local/cuda"
# ENV FORCE_CUDA="1"

RUN pip3 install torchmetrics==0.6.0
RUN pip3 install setuptools==59.5.0
RUN pip3 install --no-cache-dir pillow
RUN pip3 install --no-cache-dir cmake
RUN pip3 install --no-cache-dir  --upgrade pip

RUN pip3 install pytest==7.0.1
RUN pip3 install gym==0.15.7

# RUN pip3 install --no-cache-dir -r requirements.txt

# RUN pip3 install taming-transformers

# RUN pip3 install --no-cache-dir tensorflow==1.14

# RUN pip3 uninstall torch -y
# RUN pip3 uninstall torchvision -y
# RUN pip3 uninstall torchaudio -y

RUN apt-get install git -y


# RUN pip3 install "git+https://github.com/affanmehmood/taming-transformers"



# RUN pip3 install pillow==9.0.1
RUN git clone https://github.com/affanmehmood/taming-transformers --quiet
RUN pip3 install -e taming-transformers --quiet
RUN pip3 install omegaconf einops pytorch-lightning test-tube transformers kornia -e git+https://github.com/openai/CLIP.git@main#egg=clip

RUN git clone https://github.com/affanmehmood/Stable-textual-inversion_win.git
RUN pip3 install -e Stable-textual-inversion_win --quiet

# RUN pip3 install -e Stable-textual-inversion_win --quiet


# RUN pip3 install torch torchvision torchaudio

# RUN cd DALLE-pytorch
# RUN python3 /DALLE-pytorch/setup.py install
RUN cd ..
# RUN rm requirements.txt

COPY . /
WORKDIR /

ENTRYPOINT [ "python3" ]

CMD ["infer.py" ]
