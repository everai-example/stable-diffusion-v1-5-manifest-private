# the following basic images including：
#     diffusers~=0.27.2
#     transformers~=4.39.3
#     torch~=2.2.2
#     accelerate~=0.30.1
#     protobuf~=5.26.1
#     sentencepiece~=0.2.0
#     numpy~=1.26.4
#     Flask~=3.0.3
#     Werkzeug~=3.0.3
######################################
#FROM everai2024/diffusers-pytorch-gpu:v0.0.1
FROM quay.io/everai2024/diffusers-pytorch-gpu:v0.0.1

WORKDIR /workspace

RUN mkdir -p $WORKDIR/volume

COPY app.py requirements.txt ./

RUN  pip install -r requirements.txt

CMD ["python", "app.py"]
