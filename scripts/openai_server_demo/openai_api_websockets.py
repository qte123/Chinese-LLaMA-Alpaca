import argparse
import os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import websockets
from typing import List
import asyncio
import uvicorn
from threading import Thread
from sse_starlette.sse import EventSourceResponse
import logging
import json
import time
import gc
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--load_in_8bit',action='store_true', help='Load the model in 8bit mode')
parser.add_argument('--load_in_4bit',action='store_true', help='Load the model in 4bit mode')
parser.add_argument('--only_cpu',action='store_true',help='Only use CPU for inference')
parser.add_argument('--alpha',type=str,default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("Quantization is unavailable on CPU.")
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
import torch.nn.functional as F
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
from peft import PeftModel

import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from attn_and_long_ctx_patches import apply_attention_patch, apply_ntk_scaling_patch

apply_attention_patch(use_memory_efficient_attention=True)
apply_ntk_scaling_patch(args.alpha)

from openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
)

load_type = torch.float16
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device("cpu")
if args.tokenizer_path is None:
    args.tokenizer_path = args.lora_model
    if args.lora_model is None:
        args.tokenizer_path = args.base_model
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)
if args.load_in_4bit or args.load_in_8bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=load_type,
    )
base_model = LlamaForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=load_type,
    low_cpu_mem_usage=True,
    device_map='auto' if not args.only_cpu else None,
    load_in_4bit=args.load_in_4bit,
    load_in_8bit=args.load_in_8bit,
    quantization_config=quantization_config if (args.load_in_4bit or args.load_in_8bit) else None
)

model_vocab_size = base_model.get_input_embeddings().weight.size(0)
tokenizer_vocab_size = len(tokenizer)
print(f"Vocab of the base model: {model_vocab_size}")
print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
if model_vocab_size != tokenizer_vocab_size:
    print("Resize model embeddings to fit tokenizer")
    base_model.resize_token_embeddings(tokenizer_vocab_size)
if args.lora_model is not None:
    print("loading peft model")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_model,
        torch_dtype=load_type,
        device_map="auto",
    )
else:
    model = base_model

if device == torch.device("cpu"):
    model.float()

model.eval()

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE_WITH_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n" "{system_prompt}\n" "<</SYS>>\n\n" "{instruction} [/INST]"
)

TEMPLATE_WITHOUT_SYSTEM_PROMPT = "[INST] {instruction} [/INST]"


def generate_prompt(
    instruction, response="", with_system_prompt=True, system_prompt=None
):
    if with_system_prompt is True:
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map(
            {"instruction": instruction, "system_prompt": system_prompt}
        )
    else:
        prompt = TEMPLATE_WITHOUT_SYSTEM_PROMPT.format_map({"instruction": instruction})
    if len(response) > 0:
        prompt += " " + response
    return prompt


def generate_completion_prompt(instruction: str):
    """Generate prompt for completion"""
    return generate_prompt(instruction, response="", with_system_prompt=True)


def generate_chat_prompt(messages: list):
    """Generate prompt for chat completion"""

    system_msg = None
    for msg in messages:
        if msg.role == "system":
            system_msg = msg.content
    prompt = ""
    is_first_user_content = True
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            if is_first_user_content is True:
                prompt += generate_prompt(
                    msg.content, with_system_prompt=True, system_prompt=system_msg
                )
                is_first_user_content = False
            else:
                prompt += "<s>" + generate_prompt(msg.content, with_system_prompt=False)
        if msg.role == "assistant":
            prompt += f" {msg.content}" + "</s>"
    return prompt

def calculate_token_efficiency(chunks, elapsed_time):
    # 计算输出token数量
    output_token_count = len(chunks)
    # 计算token效率
    token_efficiency = output_token_count / elapsed_time
    return token_efficiency

logging.basicConfig(level=logging.INFO)  # 设置日志级别为 INFO

async def get_chunk(streamer,websocket,model_id="chinese-llama-alpaca-2"):
     average_start_time = time.time()
     new_chunks = [] # 添加日志记录
     new_chunk = ""  # 或其他适当的初始值 
     for new_text in streamer:
            single_start_time = time.time()
            choice_data = ChatCompletionResponseStreamChoice(
                index=0, delta=DeltaMessage(content=new_text), finish_reason=None
            )
            chunk = ChatCompletionResponse(
                model=model_id, choices=[choice_data], object="chat.completion.chunk"
            )
            end_time = time.time()  # 获取结束时间戳
            average_elapsed_time = end_time - average_start_time  # 计算平均输出时间
            single_elapsed_time = end_time - single_start_time  # 计算单个输出时间
            average_token_efficiency = calculate_token_efficiency(new_chunks, average_elapsed_time)  # 计算平均输出效率
            single_token_efficiency = calculate_token_efficiency(new_chunk, single_elapsed_time)  
            chunk_data=chunk.json(exclude_unset=True, ensure_ascii=False)
            chunk_json = "{}".format(chunk_data)
            logging.info(f"Generated chunk: {chunk_json}")
            chunk_dictionary = json.loads(chunk_data)  
            if "content" not in chunk_dictionary["choices"][0]["delta"]:
                new_chunk=""
            else:
                new_chunk=chunk_dictionary["choices"][0]["delta"]["content"]
            new_chunks.append(new_chunk)

            message = {"type": "text", "chunk": new_chunk, "chunks": new_chunks, "token_length": len(
                new_chunks), "average_elapsed_time": average_elapsed_time, "average_token_efficiency": average_token_efficiency, "single_elapsed_time": single_elapsed_time, "single_token_efficiency": single_token_efficiency}
            logging.info(f"Generated chunk: {message}")
            await websocket.send_json(message)

async def stream_predict(
    input,
    websocket: WebSocket,
    max_new_tokens=128,
    top_p=0.75,
    temperature=0.1,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.0,
    do_sample=True,
    model_id="chinese-llama-alpaca-2",
    **kwargs,
):
    try:
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            choices=[choice_data],
            object="chat.completion.chunk",
        )
        chunk_data=chunk.json(exclude_unset=True, ensure_ascii=False)
        chunk_json = "{}".format(chunk_data)
        # 添加日志记录
        logging.info(f"Generated chunk: {chunk_json}")
        # await websocket.send_json(chunk_data)
        if isinstance(input, str):
            prompt = generate_completion_prompt(input)
        else:
            prompt = generate_chat_prompt(input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            **kwargs,
        )

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            streamer=streamer,
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=float(repetition_penalty),
        )
        # Thread(target=model.generate, kwargs=generation_kwargs).start()
        await get_chunk(streamer,websocket)
        generated_chunk = await asyncio.to_thread(model.generate, **generation_kwargs)
        
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(), finish_reason="stop"
        )
        chunk = ChatCompletionResponse(
            model=model_id, choices=[choice_data], object="chat.completion.chunk"
        )
        chunk_data=chunk.json(exclude_unset=True, ensure_ascii=False)
        chunk_json = "{}".format(chunk_data)
        # 添加日志记录
        logging.info(f"Generated chunk: {chunk_json}")
        # await websocket.send_json(chunk_data)
    except websockets.ConnectionClosedOK:
        print(f"WebSocket connection closed")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 使用一个列表来管理WebSocket连接
websocket_connections: List[WebSocket] = []
# 创建 WebSocket 聊天路由

@app.websocket("/v1/chat/")
async def create_chat_completion(websocket: WebSocket):
    await websocket.accept()
    websocket_connections.append(websocket)
    """Creates a completion for the chat message"""
    try:
        request_data = await websocket.receive_json()
        logging.info(
            f"Received message {request_data}")  # 记录接收的消息
        request = ChatCompletionRequest(**request_data)
        msgs=request.messages
        if isinstance(msgs, str):
            msgs = [ChatMessage(role="user", content=msgs)]
        else:
            msgs = [ChatMessage(role=x["role"], content=x["content"]) for x in msgs]
        await stream_predict(
                input=msgs,
                websocket=websocket,
                max_new_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                temperature=request.temperature,
                num_beams=request.num_beams,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
        )
    except Exception as e:
        print(f"WebSocket error: {e}")
        # 手动执行垃圾回收
        # 执行一些操作后清除缓存
        torch.cuda.empty_cache()
        gc.collect()
    finally:
        await websocket.close()  # 使用await等待连接关闭操作完成
        # 连接关闭时从列表中移除
        websocket_connections.remove(websocket)
        # 执行一些操作后清除缓存
        torch.cuda.empty_cache()
        gc.collect()
        pass

# 启动FastAPI应用
if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0"
    post = 19324
    uvicorn.run(app, host=host, port=post)
