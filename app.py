from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch

import flask
from flask import Response
from flask import Flask

import os
import io
import PIL
from io import BytesIO

MODEL_NAME = 'runwayml/stable-diffusion-v1-5'
HUGGINGFACE_SECRET_NAME = os.environ.get("HF_TOKEN")

app = Flask(__name__)

VOLUME_NAME = 'volume'

txt2img_pipe = None
img2img_pipe = None

def prepare_model():
    volume = VOLUME_NAME
    huggingface_token = HUGGINGFACE_SECRET_NAME

    model_dir = volume

    global txt2img_pipe
    global img2img_pipe

    txt2img_pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME,
                                                        token=huggingface_token,
                                                        cache_dir=model_dir,
                                                        revision="fp16", 
                                                        torch_dtype=torch.float16, 
                                                        low_cpu_mem_usage=False,
                                                        local_files_only=True
                                                        )
    
    # The self.components property can be useful to run different pipelines with the same weights and configurations without reallocating additional memory.
    # If you just want to use img2img pipeline, you should use StableDiffusionImg2ImgPipeline.from_pretrained below.
    img2img_pipe = StableDiffusionImg2ImgPipeline(**txt2img_pipe.components)
    #image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_NAME,
    #                                                            token=huggingface_token,
    #                                                            cache_dir=model_dir,
    #                                                            revision="fp16",
    #                                                            torch_dtype=torch.float16, 
    #                                                            low_cpu_mem_usage=False,
    #                                                            local_files_only=True
    #                                                            )
      
    if torch.cuda.is_available():
        txt2img_pipe.to("cuda")
        img2img_pipe.to("cuda")
    elif torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        txt2img_pipe.to(mps_device)
        img2img_pipe.to(mps_device)

# service entrypoint
# api service url looks https://everai.expvent.com/api/routes/v1/default/stable-diffusion-v1-5/txt2img
# for test local url is http://127.0.0.1:8866/txt2img
@app.route('/txt2img', methods=['GET','POST'])
def txt2img():    
    if flask.request.method == 'POST':
        data = flask.request.json
        prompt = data['prompt']
    else:
        prompt = flask.request.args["prompt"]

    pipe_out = txt2img_pipe(prompt)

    image_obj = pipe_out.images[0]

    byte_stream = io.BytesIO()
    image_obj.save(byte_stream, format="PNG")

    return Response(byte_stream.getvalue(), mimetype="image/png")

# service entrypoint
# api service url looks https://everai.expvent.com/api/routes/v1/default/stable-diffusion-v1-5/img2img
# for test local url is http://127.0.0.1:8866/img2img
@app.route('/img2img', methods=['POST'])
def img2img():        
    f = flask.request.files['file']
    img = f.read()

    prompt = flask.request.form['prompt']
    
    init_image = PIL.Image.open(BytesIO(img)).convert("RGB")
    init_image = init_image.resize((768, 512))

    pipe_out = img2img_pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5)

    image_obj = pipe_out.images[0]

    byte_stream = io.BytesIO()
    image_obj.save(byte_stream, format="JPEG")

    return Response(byte_stream.getvalue(), mimetype="image/jpg")

@app.route('/healthy-check', methods=['GET'])
def healthy():
    resp = 'container is ready'
    return resp

if __name__ == '__main__':
    prepare_model()
    app.run(host="0.0.0.0", debug=False, port=8866)

