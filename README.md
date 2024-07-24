#  Stable Diffusion 1.5 by manifest yaml file

In this example, we can implement an AIGC(AI generated content) online text-to-image and image-to-image service based on the `Stable Diffusion 1.5` model on the [EverAI](https://everai.expvent.com) platform by defining a manifest yaml file. This method allows you to deploy your application to the EverAI platform without importing EverAI SDK code into your existing business code.  

## Login EVERAI CLI

Create a directory for your app firstly. In your app directory, you should login by token you got in [EverAI](https://everai.expvent.com).  
    
```bash
everai login --token <your token>
```

## Prepare volume

Create a data volume to store model files.  

```bash
everai volume create models--runwayml--stable-diffusion-v1-5
```

You can get the local path of the volume `models--runwayml--stable-diffusion-v1-5` through the `everai volume get` command. After entering the local path of the volume, you can copy your model files to that path.

Use the `everai volume push` command to push the model files in the local environment of the data volume `models--runwayml--stable-diffusion-v1-5` to the cloud.

```bash
everai volume push models--runwayml--stable-diffusion-v1-5
```

In the current application directory, create a soft link to mount the local path of the volume `models--runwayml--stable-diffusion-v1-5`.  

```bash
ln -s /root/.cache/everai/volumes/Xo6zoFc4986CrD7dYuNrwr volume
```

## Write your app code in python

### Load model

There is an example code in [app.py](https://github.com/everai-example/stable-diffusion-v1-5-manifest-private/blob/main/app.py).  

When you use `python run` to debug the sample code to load the model in the local environment, the model file will be read from the private volume `models--runwayml--stable-diffusion-v1-5` in the local environment. If your local debugging environment has GPU resources(Nvidia GPU or GPU for MacOS devices with Metal programming framework), running `python run` will be successful.

```python
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch

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
```

### Generate inference service
#### Text generates image

Aftering loading `Stable Diffusion 1.5` model, now you can write your Python code that uses `flask` to implement the inference online text-to-image service of AIGC(AI generated content).  
   

```python
import flask
from flask import Response

import io

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
```

#### Image generates image
Now you can write your Python code that uses `flask` to implement the inference online image-to-image service of AIGC(AI generated content).  

```python
import PIL
from io import BytesIO

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
```

### Generate readinessProbe service

If readinessProbe is set up, there are no any request be route to this worker before probe status is ready(status code is `200`), otherwise (readinessProbe is not set up), everai platform will route client reqeust to this worker when container is ready, even model files have not been loaded into memory of GPU yet. 

HTTP `get` and `post` probe is the only supported methods now.

```python
@app.route('/healthy-check', methods=['GET'])
def healthy():
    resp = 'service is ready'
    return resp

if __name__ == '__main__':
    prepare_model()
    app.run(host="0.0.0.0", debug=False, port=8866)
```

## Build image
This step will build the container image using `Dockerfile`.    

There is an example code in [Dockerfile](https://github.com/everai-example/stable-diffusion-v1-5-manifest-private/blob/main/Dockerfile).   

You can choose the public image registry to store application image, such as [quay.io](https://quay.io/), [Docker Hub](https://hub.docker.com/), [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry), [Google Container Registry](https://cloud.google.com/artifact-registry), etc. If you have a self-built image registry and the image can be accessed on the Internet, you can also use it.  

The dependence of this step is docker installed on your machine. It is recommended to use `docker buildx` to build Docker image that support multi-platform architecture, and push the packaged image to your specified registry.

## Create secrets
Secrets are a secure way to add credentials and other sensitive information to the containers your functions run in.  

This step is optional, depending on whether the model and Docker image require security certification.  

You can create and edit secrets on [EverAI](https://everai.expvent.com), or programmatically from Python code.  

In this case, we will create one secret for [quay.io](https://quay.io/). 
```bash
everai secret create your-quay-io-secret-name \
  --from-literal username=<your username> \
  --from-literal password=<your password>
```

## Create configmap
>Optional, but you can use configmap for adjust autoscaling policy after deploying the image. 
```shell
ever configmap create sd15-configmap \ 
  --from-literal min_workers=1 \
  --from-literal max_workers=5 \
  --from-literal max_queue_size=2 \
  --from-literal scale_up_step=1 \
  --from-literal max_idle_time=60
```

## Define manifest file
The manifest file defines various information required to create an EverAI application, including application name, image name, key information, data volume information, etc.  

There is an example code in [app.yaml](https://github.com/everai-example/stable-diffusion-v1-5-manifest-private/blob/main/app.yaml).

## Deploy image
The final step is to deploy your app to everai and keep it running.  

```bash
everai app create --from-file app.yaml
```

After running `everai app list`, you can see the result similar to the following. Note that `CREATED_AT` uses UTC time display.  

If your app's status is `DEPLOYED`, and the number of ready worker containers is equal to the expected number of worker containers, which is `1/1`, it means that your app is deployed successfully.  

```bash
NAME                   NAMESPACE    STATUS    WORKERS    CREATED_AT
---------------------  -----------  --------  ---------  ------------------------
stable-diffusion-v1-5  default      DEPLOYED  1/1        2024-06-19T05:16:05+0000
```

When your app is deployed, you can use `curl` to execute the following request to test your deployed text-to-image code, A picture generated by the `Stable Diffusion 1.5` model will be downloaded in the current directory on the console.  

```bash
curl -X POST -d '{"prompt": "a photo of a cat on the boat"}' -H 'Content-Type: application/json' -H'Authorization: Bearer <your_token>' -o test.png https://everai.expvent.com/api/routes/v1/<your namespace>/<your app name>/txt2img
```

Open the picture and you can see the following effect.  

<img src="https://expvent.com.cn:1111/evfiles/v1/expvent/public/everai-documentation/demo-cat.png" width = "512" />  

You also can use `curl` to execute the following request to test your deployed image-to-image code.  

Before using `curl` to request, you need to download `sketch-mountains-input.jpg` to your local directory, execute `curl` in the directory of the console terminal, and a new image based on this original image of `sketch-mountains-input.jpg` and `prompt` will be generated by the large model `Stable Diffusion 1.5`.  

```bash
curl -X POST -F 'file=@sketch-mountains-input.jpg' -F "prompt=A fantasy landscape, trending on artstation" -H'Authorization: Bearer <your_token>' -o test.jpg https://everai.expvent.com/api/routes/v1/<your namespace>/<your app name>/img2img
```

An example of the original image is shown below.  

<img src="https://expvent.com.cn:1111/evfiles/v1/expvent/public/everai-documentation/sketch-mountains-input.jpg" width = "512" />

Open the new picture and you can see the following effect.  

<img src="https://expvent.com.cn:1111/evfiles/v1/expvent/public/everai-documentation/img2img-output.jpg" width = "512" />