```shell
# websocket版 不带页面
python scripts/inference/gradio_demo_websockets.py --base_model /path/to/base_model --gpus 0,1,2,3
# api版 带页面
python scripts/inference/gradio_demo.py --base_model /path/to/base_model --gpus 0,1,2,3
# api 不带页面
python scripts/inference/inference_hf.py \
    --base_model /path/to/base_model \
    --with_prompt \
    --interactive \
    --gpus 0,1,2,3
```

