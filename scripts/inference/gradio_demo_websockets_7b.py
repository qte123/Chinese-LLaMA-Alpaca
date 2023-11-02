import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    BitsAndBytesConfig
)
from pydantic import BaseModel
import gradio as gr
import argparse
import os
from queue import Queue
from threading import Thread
import traceback
import gc
import json
import requests
from typing import Iterable, List
import subprocess
import re
import logging
import time
import asyncio
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import websockets
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE_WITH_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

TEMPLATE_WITHOUT_SYSTEM_PROMPT = "[INST] {instruction} [/INST]"

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--base_model',
    default=None,
    type=str,
    required=True,
    help='Base model path')
parser.add_argument('--lora_model', default=None, type=str,
                    help="If None, perform inference on the base model")
parser.add_argument(
    '--tokenizer_path',
    default=None,
    type=str,
    help='If None, lora model path or base model path will be used')
parser.add_argument(
    '--gpus',
    default="0",
    type=str,
    help='If None, cuda:0 will be used. Inference using multi-cards: --gpus=0,1,... ')
parser.add_argument('--share', default=True, help='Share gradio domain name')
parser.add_argument('--port', default=19324, type=int, help='Port of gradio demo')
parser.add_argument(
    '--max_memory',
    default=1024,
    type=int,
    help='Maximum number of input tokens (including system prompt) to keep. If exceeded, earlier history will be discarded.')
parser.add_argument(
    '--load_in_8bit',
    action='store_true',
    help='Use 8 bit quantized model')
parser.add_argument(
    '--load_in_4bit',
    action='store_true',
    help='Use 4 bit quantized model')
parser.add_argument(
    '--only_cpu',
    action='store_true',
    help='Only use CPU for inference')
parser.add_argument(
    '--alpha',
    type=str,
    default="1.0",
    help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument(
    "--use_vllm",
    action='store_true',
    help="Use vLLM as back-end LLM service.")
parser.add_argument(
    "--post_host",
    type=str,
    default="0.0.0.0",
    help="Host of vLLM service.")
parser.add_argument(
    "--post_port",
    type=int,
    default=8000,
    help="Port of vLLM service.")
args = parser.parse_args()

ENABLE_CFG_SAMPLING = True
try:
    from transformers.generation import UnbatchedClassifierFreeGuidanceLogitsProcessor
except ImportError:
    ENABLE_CFG_SAMPLING = False
    print("Install the latest transformers (commit equal or later than d533465) to enable CFG sampling.")
if args.use_vllm is True:
    print("CFG sampling is disabled when using vLLM.")
    ENABLE_CFG_SAMPLING = False

if args.only_cpu is True:
    print('only cpu')
    args.gpus = ""
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("Quantization is unavailable on CPU.")
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from attn_and_long_ctx_patches import apply_attention_patch, apply_ntk_scaling_patch
if not args.only_cpu:
    apply_attention_patch(use_memory_efficient_attention=True)
apply_ntk_scaling_patch(args.alpha)

# Set CUDA devices if available
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


# Peft library can only import after setting CUDA devices
from peft import PeftModel


# Set up the required components: model and tokenizer

def setup():
    global tokenizer, model, device, share, port, max_memory
    if args.use_vllm:
        # global share, port, max_memory
        max_memory = args.max_memory
        port = args.port
        share = args.share

        if args.lora_model is not None:
            raise ValueError("vLLM currently does not support LoRA, please merge the LoRA weights to the base model.")
        if args.load_in_8bit or args.load_in_4bit:
            raise ValueError("vLLM currently does not support quantization, please use fp16 (default) or unuse --use_vllm.")
        if args.only_cpu:
            raise ValueError("vLLM requires GPUs with compute capability not less than 7.0. If you want to run only on CPU, please unuse --use_vllm.")

        if args.tokenizer_path is None:
            args.tokenizer_path = args.base_model
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)

        print("Start launch vllm server.")
        cmd = f"python -m vllm.entrypoints.api_server \
            --model={args.base_model} \
            --tokenizer={args.tokenizer_path} \
            --tokenizer-mode=slow \
            --tensor-parallel-size={len(args.gpus.split(','))} \
            --host {args.post_host} \
            --port {args.post_port} \
            &"
        subprocess.check_call(cmd, shell=True)
    else:
        max_memory = args.max_memory
        port = args.port
        share = args.share
        load_type = torch.float16
        if torch.cuda.is_available():
            device = torch.device(0)
        else:
            device = torch.device('cpu')
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
            device_map='auto',
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
                device_map='auto',
            ).half()
        else:
            model = base_model

        if device == torch.device('cpu'):
            model.float()

        model.eval()


# Reset the user input
def reset_user_input():
    return gr.update(value='')


# Reset the state
def reset_state():
    return []


def generate_prompt(instruction, response="", with_system_prompt=True, system_prompt=DEFAULT_SYSTEM_PROMPT):
    if with_system_prompt is True:
        prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map({'instruction': instruction,'system_prompt': system_prompt})
    else:
        prompt = TEMPLATE_WITHOUT_SYSTEM_PROMPT.format_map({'instruction': instruction})
    if len(response)>0:
        prompt += " " + response
    return prompt


# User interaction function for chat
def user(user_message, history):
    return gr.update(value="", interactive=False), history + \
        [[user_message, None]]


class Stream(StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).

    Adapted from: https://stackoverflow.com/a/9969000
    """
    def __init__(self, func, kwargs=None, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs or {}
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except Exception:
                traceback.print_exc()

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()


def clear_torch_cache():
    gc.collect()
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      top_p: float = 0.9,
                      top_k: int = 40,
                      temperature: float = 0.2,
                      max_tokens: int = 512,
                      presence_penalty: float = 1.0,
                      use_beam_search: bool = False,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "top_p": 1 if use_beam_search else top_p,
        "top_k": -1 if use_beam_search else top_k,
        "temperature": 0 if use_beam_search else temperature,
        "max_tokens": max_tokens,
        "use_beam_search": use_beam_search,
        "best_of": 5 if use_beam_search else n,
        "presence_penalty": presence_penalty,
        "stream": stream,
    }
    print(pload)

    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response

logging.basicConfig(level=logging.INFO)  # 设置日志级别为 INFO


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            logging.info(f"Generated chunk: {output}")
            yield output

###################分割线##############################
# 计算token效率
def calculate_token_efficiency(chunks, elapsed_time):
    # 计算输出token数量
    output_token_count = len(chunks)
    # 计算token效率
    token_efficiency = output_token_count / elapsed_time
    return token_efficiency

#计算token的平均效率
def calculate_token_aver_efficiency(token_efficiency_list):
    token_efficiency_arr = np.array(token_efficiency_list)
    number = token_efficiency_arr.size  # 使用数组的size属性获取长度
    num_efficiency = np.sum(token_efficiency_arr)
    token_aver_efficiency = np.mean(token_efficiency_arr)
    return token_aver_efficiency

# 计算TPS
def get_token_per_second(chunks, elapsed_time):
    # 计算输出token数量
    output_token_count = len(chunks)
    # 计算TPS
    token_per_second = output_token_count / elapsed_time
    return token_per_second

# 计算WPS
def get_word_per_second(chunks, elapsed_time):
    # 计算总单词长度
    total_words_len = sum(len(chunk) for chunk in chunks)
    # 计算单词每秒数
    word_per_second = total_words_len / elapsed_time
    return word_per_second


# 数据处理和曲线绘制
def create_plot(outputs, max_token, name,is_gpu=True):
    # 指定 data 文件夹的路径
    data_folder_plt = "data/plt"
    # 生成带有时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"output_{timestamp}"
    outputs = pd.DataFrame(outputs)
    # 计算除第一行外的数据行数量
    list_length = len(outputs)

    # 根据数据行数量选择 token_length 的长度
    token_length = outputs["token_length"][0 : list_length + 1]

    # 提取数据列
    spend_times = outputs["spend_time"][0 : list_length + 1]
    token_per_second = outputs["token_per_second"][0 : list_length + 1]
    word_per_second = outputs["word_per_second"][0 : list_length + 1]

    # 获取数据中的最小值和最大值
    min_token_length = np.min(token_length)
    max_token_length = np.max(token_length)

    # 设置x轴的范围
    x_length = [min_token_length, max_token]
    # 设置y轴的范围
    y_time_length = [0,150]
    y_tps_length=[0,16]
    y_wps_length=[0,50]

    # 创建图像
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # 指定需要标注的 x 坐标
    x_ticks = np.arange(0, max_token + 1, 100)
    #指定需要标注的x的坐标
    x_ticks_ = np.arange(0, max_token + 1, 200)
    # 使用 numpy 的 isin 函数筛选出满足条件的数据点下标
    indices = np.isin(token_length, x_ticks)
    selected_x = token_length[indices]
    indices_ = np.isin(token_length, x_ticks_)
    selected_x_ = token_length[indices_]
    # 绘制 "spend_time" 的曲线图
    ax1.plot(token_length, spend_times, marker="", linestyle="-", color="b")
    # 获取满足条件的 x 坐标及对应的 y 坐标
    selected_y_time = spend_times[indices]
    selected_y_time_ = spend_times[indices_]
    #设置表格
    x_tables = np.append(selected_x, [token_length.iloc[-1]])
    y_tables_time = np.append(selected_y_time, [spend_times.iloc[-1]])
    data1 = {
    "Token Length": x_tables,
    "Spend Time": y_tables_time
    }
    df1 = pd.DataFrame(data1)
    for x, y in zip(selected_x, selected_y_time):
        ax1.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        ax1.scatter(x, y, color="black")  # 添加散点图，用红色表示
    # 标注曲线的最后一个点
    ax1.annotate(
        f"{spend_times.iloc[-1]:.2f}",
        (token_length.iloc[-1], spend_times.iloc[-1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )
    
    # 显示表格数据 1
    cell_text1 = []
    for row in range(len(df1)):
        rounded_values = df1.iloc[row].round(2)  # 对每一行数据四舍五入保留两位小数
        cell_text1.append(rounded_values.values)
    ax1.table(cellText=cell_text1, colLabels=df1.columns, cellLoc='center', loc='bottom', bbox=[0, -0.55, 1, 0.3])
    
    ax1.scatter(token_length.iloc[-1], spend_times.iloc[-1], color="black")
    ax1.set_xlabel("Token Length")
    ax1.set_ylabel("Spend Time")
    ax1.set_title("Spend Time by Token Length")
    ax1.grid(True)
    # 设置 x 轴范围
    ax1.set_xlim(x_length)
    # 设置 x 轴刻度的间隔
    ax1.set_xticks(x_ticks)
    ax1.set_ylim(y_time_length)

    # 绘制 "token_efficiency" 的曲线图
    ax2.plot(token_length, token_per_second, marker="", linestyle="-", color="r")
    selected_y_efficiencies = token_per_second[indices]
    selected_y_efficiencies_ = token_per_second[indices_]
    y_tables_efficiencies=np.append(selected_y_efficiencies,[token_per_second.iloc[-1]])
    data2 = {
    "Token Length": x_tables,
    "Token Efficiency": y_tables_efficiencies
    }
    df2=pd.DataFrame(data2)
    for x, y in zip(selected_x, selected_y_efficiencies):
        ax2.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        ax2.scatter(x, y, color="black")  # 添加散点图，用红色表示
    ax2.annotate(
    f"{token_per_second.iloc[-1]:.2f}",
    (token_length.iloc[-1], token_per_second.iloc[-1]),
    textcoords="offset points",
    xytext=(0, 10),
    ha="center",
    )
    
    # 显示表格数据 2
    cell_text2 = []
    for row in range(len(df2)):
        rounded_values = df2.iloc[row].round(2)  # 对每一行数据四舍五入保留两位小数
        cell_text2.append(rounded_values.values)
    ax2.table(cellText=cell_text2, colLabels=df2.columns, cellLoc='center', loc='bottom', bbox=[0, -0.55, 1, 0.3])

    ax2.scatter(token_length.iloc[-1], token_per_second.iloc[-1], color="black")
    ax2.set_xlabel("Token Length")
    ax2.set_ylabel("Token per second (TPS)")
    ax2.set_title("Token per second (TPS) by Token Length")
    ax2.grid(True)
    # 设置 x 轴范围
    ax2.set_xlim(x_length)
    # 设置 x 轴刻度的间隔
    ax2.set_xticks(x_ticks)
    ax2.set_ylim(y_tps_length)

    # 绘制 "token_aver_efficiency" 的曲线图
    ax3.plot(token_length, word_per_second, marker="", linestyle="-", color="g")
    selected_y_aver_efficiencies = word_per_second[indices]
    selected_y_aver_efficiencies_ = word_per_second[indices_]
    #设置表格
    y_tables_aver_efficiencies=np.append(selected_y_aver_efficiencies,[word_per_second.iloc[-1]])
    data3 = {
    "Token Length": x_tables,
    "Token Average Efficiency": y_tables_aver_efficiencies
    }
    df3 = pd.DataFrame(data3)
    for x, y in zip(selected_x, selected_y_aver_efficiencies):
        ax3.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        ax3.scatter(x, y, color="black")  # 添加散点图，用红色表示
    ax3.annotate(
    f"{word_per_second.iloc[-1]:.2f}",
    (token_length.iloc[-1], word_per_second.iloc[-1]),
    textcoords="offset points",
    xytext=(0, 10),
    ha="center",
    )

    # 显示表格数据 3
    cell_text3 = []
    for row in range(len(df3)):
        rounded_values = df3.iloc[row].round(2)  # 对每一行数据四舍五入保留两位小数
        cell_text3.append(rounded_values.values)
    ax3.table(cellText=cell_text3, colLabels=df3.columns, cellLoc='center', loc='bottom', bbox=[0, -0.55, 1, 0.3])

    ax3.scatter(token_length.iloc[-1], word_per_second.iloc[-1], color="black")
    ax3.set_xlabel("Token Length")
    ax3.set_ylabel("Word per second (WPS)")
    ax3.set_title("Word per second (WPS) by Token Length")
    ax3.grid(True)
    # 设置 x 轴范围
    ax3.set_xlim(x_length)
    # 设置 x 轴刻度的间隔
    ax3.set_xticks(x_ticks)
    ax3.set_ylim(y_wps_length)
    

    plt.subplots_adjust(wspace=0.4)
    # 设置总标题
    if is_gpu:
        num_gpus = torch.cuda.device_count()
        fig.suptitle(
        f"{name}(GPUS{num_gpus}) Generation Efficiency",
        fontsize=16,
        fontweight="bold",
    )
    else:
        fig.suptitle(
            f"{name}(ONLY CPU) Generation Efficiency",
            fontsize=16,
            fontweight="bold",
        )

    # 生成带有时间戳的文件名
    png_filename = f"{name}_{filename}_gpus{num_gpus}.png"
    png_filepath = os.path.join(data_folder_plt, png_filename)
    plt.tight_layout()
    # 保存 PNG 文件
    plt.savefig(png_filepath)
    # 关闭图形
    plt.close(fig)
    # 显示图像
    # plt.show()

# 定义请求模型
class ChatRequest(BaseModel):
    user_prompt: str
    system_prompt: str = "You are a helpful assistant"

# Perform prediction based on the user input and history
@torch.no_grad()
async def predict(
    # history,
    websocket: WebSocket,
    request:ChatRequest,
    negative_prompt="",
    max_new_tokens=2048,
    top_p=0.9,
    temperature=0.2,
    top_k=40,
    do_sample=True,
    repetition_penalty=1.1,
    guidance_scale=1.0,
    presence_penalty=0.0,
):
    system_prompt=request.system_prompt
    user_prompt=request.user_prompt
    if len(system_prompt) == 0:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    is_first_user_content = True
    if is_first_user_content is True:
        input = user_prompt
        prompt = generate_prompt(input,response="", with_system_prompt=True, system_prompt=system_prompt)
        is_first_user_content = False

    else:
        input = user_prompt
        prompt = generate_prompt(input, response="", with_system_prompt=True, system_prompt=system_prompt)+'</s>'

    if args.use_vllm:
        generate_params = {
            'max_tokens': max_new_tokens,
            'top_p': top_p,
            'temperature': temperature,
            'top_k': top_k,
            "use_beam_search": not do_sample,
            'presence_penalty': presence_penalty,
        }

        api_url = f"http://{args.post_host}:{args.post_port}/generate"


        response = post_http_request(prompt, api_url, **generate_params, stream=True)

        # for h in get_streaming_response(response):
        #     for line in h:
        #         line = line.replace(prompt, '')
        #         history[-1][1] = line
        #         yield history

    else:
        try:
            average_start_time = time.time()
            negative_text = None
            if len(negative_prompt) != 0:
                negative_text = re.sub(r"<<SYS>>\n(.*)\n<</SYS>>", f"<<SYS>>\n{negative_prompt}\n<</SYS>>", prompt)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            if negative_text is None:
                negative_prompt_ids = None
                negative_prompt_attention_mask = None
            else:
                negative_inputs = tokenizer(negative_text,return_tensors="pt")
                negative_prompt_ids = negative_inputs["input_ids"].to(device)
                negative_prompt_attention_mask = negative_inputs["attention_mask"].to(device)
            generate_params = {
                'input_ids': input_ids,
                'max_new_tokens': max_new_tokens,
                'top_p': top_p,
                'temperature': temperature,
                'top_k': top_k,
                'do_sample': do_sample,
                'repetition_penalty': repetition_penalty,
            }
            if ENABLE_CFG_SAMPLING is True:
                generate_params['guidance_scale'] = guidance_scale
                generate_params['negative_prompt_ids'] = negative_prompt_ids
                generate_params['negative_prompt_attention_mask'] = negative_prompt_attention_mask

            def generate_with_callback(callback=None, **kwargs):
                if 'stopping_criteria' in kwargs:
                    kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                else:
                    kwargs['stopping_criteria'] = [Stream(callback_func=callback)]
                clear_torch_cache()
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, kwargs, callback=None)

            chunk=""
            chunks=[]
            old_tokens = "" # 用于保存先前生成的 tokens
            character_length=0
            token_efficiency_list=[]
            outputs=[]
            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # logging.info(f"Generated chunk: {output}")
                    next_token_ids = output[len(input_ids[0]):]
                    if next_token_ids[0] == tokenizer.eos_token_id:
                        break
                    
                    new_tokens = tokenizer.decode(
                        next_token_ids, skip_special_tokens=True)
                    # logging.info(f"new tokens: {new_tokens}")
                    # logging.info(f"Generated chunk: {chunk}")
                    if isinstance(tokenizer, LlamaTokenizer) and len(next_token_ids) > 0:
                        if tokenizer.convert_ids_to_tokens(int(next_token_ids[0])).startswith('▁'):
                            new_tokens = ' ' + new_tokens
                    # logging.info(f"new tokens: {new_tokens}")
                    
                    # 计算old_token与new_token的更新的字符串
                    old_len=len(old_tokens)
                    chunk=new_tokens[old_len:]
                    end_time = time.time()  # 获取结束时间戳
                    chunks.append(chunk)
                    logging.info(f"changes tokens: {chunk}")

                    
                    token_length=len(chunks)
                    character_length+=len(chunk)
                    spend_time = end_time - average_start_time  # 计算平均输出时间
                    # single_elapsed_time = end_time - single_start_time  # 计算单个输出时间
                    token_efficiency = calculate_token_efficiency(
                    chunks, spend_time)  # 计算平均token输出效率
                    character_efficiency = calculate_token_efficiency(
                    "".join(chunks), spend_time)  # 计算单个字符输出效率
                    token_efficiency_list.append(token_efficiency)
                    token_aver_efficiency=calculate_token_aver_efficiency(token_efficiency_list)
                    token_per_second=get_token_per_second(chunks, spend_time)
                    word_per_second=get_word_per_second(chunks, spend_time)
                    # 逐个chunk发送
                    message = {
                        "type": "text", 
                        "user_prompt": user_prompt, 
                        "system_prompt": system_prompt, 
                        "chunk": chunk, 
                        "chunks": chunks, 
                        "token_length": token_length, 
                        "character_length":character_length,
                        "spend_time": spend_time, 
                        "token_efficiency": token_efficiency, 
                        "character_efficiency": character_efficiency
                    }
                    output={
                        "token_length": token_length, 
                        "spend_time": spend_time,
                        "token_per_second": token_per_second, 
                        "word_per_second":word_per_second
                    }
                    outputs.append(output)
                    # 使用 asyncio.sleep 等待一段时间，模拟生成 chunk 的时间
                    await asyncio.sleep(0.1)  # 此处的 sleep 时间根据您的生成速度来调整
                    await websocket.send_json(message)

                    # 更新先前 tokens 以备下一次循环使用
                    old_tokens = new_tokens
                    # logging.info(f"old tokens: {old_tokens}")
                    # history[-1][1] = new_tokens
                    # yield history
                    if len(next_token_ids) >= max_new_tokens:
                        break
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:       
                create_plot(outputs,1000,'Chinese-Alpaca-2-7B-16K',is_gpu=True)
            else:
                create_plot(outputs,1000,'Chinese-Alpaca-2-7B-16K',is_gpu=False)
        except websockets.ConnectionClosedOK:
            print(f"WebSocket connection closed")    


# Call the setup function to initialize the components
setup()

'''
# Create the Gradio interface
with gr.Blocks() as demo:
    github_banner_path = 'https://raw.githubusercontent.com/ymcui/Chinese-LLaMA-Alpaca-2/main/pics/banner.png'
    gr.HTML(f'<p align="center"><a href="https://github.com/ymcui/Chinese-LLaMA-Alpaca-2"><img src={github_banner_path} width="700"/></a></p>')
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=3):
                system_prompt_input = gr.Textbox(
                    show_label=True,
                    label="系统提示语（仅在对话开始前或清空历史后修改有效，对话过程中修改无效）",
                    placeholder=DEFAULT_SYSTEM_PROMPT,
                    lines=1).style(
                    container=True)
                negative_prompt_input = gr.Textbox(
                    show_label=True,
                    label="反向提示语（仅在对话开始前或清空历史后修改有效，对话过程中修改无效）",
                    placeholder="（可选，默认为空）",
                    lines=1,
                    visible=ENABLE_CFG_SAMPLING).style(
                    container=True)
            with gr.Column(scale=12):
                user_input = gr.Textbox(
                    show_label=True,
                    label="用户指令",
                    placeholder="Shift + Enter发送消息...",
                    lines=10).style(
                    container=True)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_new_token = gr.Slider(
                0,
                4096,
                value=512,
                step=1.0,
                label="Maximum New Token Length",
                interactive=True)
            top_p = gr.Slider(0, 1, value=0.9, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0,
                1,
                value=0.2,
                step=0.01,
                label="Temperature",
                interactive=True)
            top_k = gr.Slider(1, 40, value=40, step=1,
                              label="Top K", interactive=True)
            do_sample = gr.Checkbox(
                value=True,
                label="Do Sample",
                info="use random sample strategy",
                interactive=True)
            repetition_penalty = gr.Slider(
                1.0,
                3.0,
                value=1.1,
                step=0.1,
                label="Repetition Penalty",
                interactive=True,
                visible=False if args.use_vllm else True)
            guidance_scale = gr.Slider(
                1.0,
                3.0,
                value=1.0,
                step=0.1,
                label="Guidance Scale",
                interactive=True,
                visible=ENABLE_CFG_SAMPLING)
            presence_penalty = gr.Slider(
                -2.0,
                2.0,
                value=1.0,
                step=0.1,
                label="Presence Penalty",
                interactive=True,
                visible=True if args.use_vllm else False)

    params = [user_input, chatbot]
    predict_params = [
        chatbot,
        system_prompt_input,
        negative_prompt_input,
        max_new_token,
        top_p,
        temperature,
        top_k,
        do_sample,
        repetition_penalty,
        guidance_scale,
        presence_penalty]

    submitBtn.click(
        user,
        params,
        params,
        queue=False).then(
        predict,
        predict_params,
        chatbot).then(
            lambda: gr.update(
                interactive=True),
        None,
        [user_input],
        queue=False)

    user_input.submit(
        user,
        params,
        params,
        queue=False).then(
        predict,
        predict_params,
        chatbot).then(
            lambda: gr.update(
                interactive=True),
        None,
        [user_input],
        queue=False)

    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True)
'''
 


# # Launch the Gradio interface
# demo.queue().launch(
#     share=share,
#     inbrowser=True,
#     server_name='0.0.0.0',
#     server_port=19324)



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
        request = ChatRequest(**request_data)
        await predict (websocket,request)
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
    # 检查系统中可用的GPU数量
    num_gpus = torch.cuda.device_count()
    logging.info(num_gpus)
    import uvicorn
    host = "0.0.0.0"
    post = 19324
    uvicorn.run(app, host=host, port=post)
