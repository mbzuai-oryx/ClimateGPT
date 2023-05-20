python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path weights/ClimateGPT-en
python3 -m fastchat.serve.gradio_web_server