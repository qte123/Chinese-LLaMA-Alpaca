import argparse
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from threading import Thread
from sse_starlette.sse import EventSourceResponse
import logging
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    # 创建图像
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 指定需要标注的 x 坐标
    x_ticks = np.arange(0, max_token + 1, 100)

    # 使用 numpy 的 isin 函数筛选出满足条件的数据点下标
    indices = np.isin(token_length, x_ticks)
    selected_x = token_length[indices]

    # 绘制 "spend_time" 的曲线图
    ax1.plot(token_length, spend_times, marker="", linestyle="-", color="b")
    # 获取满足条件的 x 坐标及对应的 y 坐标
    selected_y_time = spend_times[indices]
    for x, y in zip(selected_x, selected_y_time):
        ax1.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        ax1.scatter(x, y, color="black")  # 添加散点图，用红色表示
    ax1.set_xlabel("Token Length")
    ax1.set_ylabel("Spend Time")
    ax1.set_title("Spend Time by Token Length")
    ax1.grid(True)
    # 设置 x 轴范围
    ax1.set_xlim(x_length)
    # 设置 x 轴刻度的间隔
    ax1.set_xticks(x_ticks)

    # 绘制 "token_efficiency" 的曲线图
    ax2.plot(token_length, token_per_second, marker="", linestyle="-", color="r")
    selected_y_efficiencies = token_per_second[indices]
    for x, y in zip(selected_x, selected_y_efficiencies):
        ax2.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        ax2.scatter(x, y, color="black")  # 添加散点图，用红色表示
    ax2.set_xlabel("Token Length")
    ax2.set_ylabel("Token per second (TPS)")
    ax2.set_title("Token per second (TPS) by Token Length")
    ax2.grid(True)
    # 设置 x 轴范围
    ax2.set_xlim(x_length)
    # 设置 x 轴刻度的间隔
    ax2.set_xticks(x_ticks)

    # 绘制 "token_aver_efficiency" 的曲线图
    ax3.plot(token_length, word_per_second, marker="", linestyle="-", color="g")
    selected_y_aver_efficiencies = word_per_second[indices]
    for x, y in zip(selected_x, selected_y_aver_efficiencies):
        ax3.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        ax3.scatter(x, y, color="black")  # 添加散点图，用红色表示
    ax3.set_xlabel("Token Length")
    ax3.set_ylabel("Word per second (WPS)")
    ax3.set_title("Word per second (WPS) by Token Length")
    ax3.grid(True)
    # 设置 x 轴范围
    ax3.set_xlim(x_length)
    # 设置 x 轴刻度的间隔
    ax3.set_xticks(x_ticks)

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
    # 保存 PNG 文件
    plt.savefig(png_filepath)
    # 关闭图形
    plt.close(fig)
    # 显示图像
    # plt.show()




def predict(
    input,
    max_new_tokens=1024,
    top_p=0.9,
    temperature=0.2,
    top_k=40,
    num_beams=1,
    repetition_penalty=1.1,
    do_sample=True,
    **kwargs,
):
    """
    Main inference method
    type(input) == str -> /v1/completions
    type(input) == list -> /v1/chat/completions
    """
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
    generation_config.return_dict_in_generate = True
    generation_config.output_scores = False
    generation_config.max_new_tokens = max_new_tokens
    generation_config.repetition_penalty = float(repetition_penalty)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    output = output.split("[/INST]")[-1].strip()
    return output

logging.basicConfig(level=logging.INFO)  # 设置日志级别为 INFO
def stream_predict(
    input,
    max_new_tokens=1024,
    top_p=0.75,
    temperature=0.1,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.0,
    do_sample=True,
    model_id="chinese-llama-alpaca-2",
    **kwargs,
):
    average_start_time = time.time()
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk",
    )
    chunk_json = "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    # 添加日志记录
    logging.info(f"Generated chunk: {chunk_json}")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

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
    Thread(target=model.generate, kwargs=generation_kwargs).start()
    chunk=""
    chunks=[]
    old_tokens = "" # 用于保存先前生成的 tokens
    character_length=0
    token_efficiency_list=[]
    outputs=[]
    for new_text in streamer:
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id, choices=[choice_data], object="chat.completion.chunk"
        )
        end_time = time.time()  # 获取结束时间戳
        chunks.append(chunk)
        token_length=len(chunks)
        spend_time = end_time - average_start_time  # 计算平均输出时间
        token_per_second=get_token_per_second(chunks, spend_time)
        word_per_second=get_word_per_second(chunks, spend_time)
        chunk_json = "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
        output={
                        "token_length": token_length, 
                        "spend_time": spend_time,
                        "token_per_second": token_per_second, 
                        "word_per_second":word_per_second
                    }
        outputs.append(output)
        # 添加日志记录
        logging.info(f"Generated chunk: {chunk_json}")
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    create_plot(outputs,1000,'chinese-llama-13B-16K',is_gpu=False)
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    chunk_json = "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    # 添加日志记录
    logging.info(f"Generated chunk: {chunk_json}")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield "[DONE]"


def get_embedding(input):
    """Get embedding main function"""
    with torch.no_grad():
        encoding = tokenizer(input, padding=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        model_output = model(input_ids, attention_mask, output_hidden_states=True)
        data = model_output.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
        masked_embeddings = data * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        seq_length = torch.sum(mask, dim=1)
        embedding = sum_embeddings / seq_length
        normalized_embeddings = F.normalize(embedding, p=2, dim=1)
        ret = normalized_embeddings.squeeze(0).tolist()
    return ret


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    msgs = request.messages
    if request.max_tokens is None:
        request.max_tokens=1024
    print(request)
    if isinstance(msgs, str):
        msgs = [ChatMessage(role="user", content=msgs)]
    else:
        msgs = [ChatMessage(role=x["role"], content=x["content"]) for x in msgs]
    if request.stream:
        generate = stream_predict(
            input=msgs,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        return EventSourceResponse(generate, media_type="text/event-stream")
    
    output = predict(
                input=msgs,
                max_new_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                temperature=request.temperature,
                num_beams=request.num_beams,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
    )
    print(output)
    choices = [
            ChatCompletionResponseChoice(index=i, message=msg) for i, msg in enumerate(msgs)
        ]
    choices += [
            ChatCompletionResponseChoice(
                index=len(choices), message=ChatMessage(role="assistant", content=output)
            )
    ]
    return ChatCompletionResponse(choices=choices)


@app.get("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    msgs = request.messages
    if request.max_tokens is None:
        request.max_tokens = 1024
    print(request)
    if isinstance(msgs, str):
        msgs = [ChatMessage(role="user", content=msgs)]
    else:
        msgs = [ChatMessage(role=x["role"], content=x["content"]) for x in msgs]
    if request.stream:
        generate = stream_predict(
            input=msgs,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        return EventSourceResponse(generate, media_type="text/event-stream")

    output = predict(
        input=msgs,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )
    print(output)
    choices = [
        ChatCompletionResponseChoice(index=i, message=msg) for i, msg in enumerate(msgs)
    ]
    choices += [
        ChatCompletionResponseChoice(
            index=len(choices), message=ChatMessage(role="assistant", content=output)
        )
    ]
    return ChatCompletionResponse(choices=choices)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Creates a completion"""
    output = predict(
        input=request.prompt,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )
    choices = [CompletionResponseChoice(index=0, text=output)]
    return CompletionResponse(choices=choices)


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    """Creates text embedding"""
    embedding = get_embedding(request.input)
    data = [{"object": "embedding", "embedding": embedding, "index": 0}]
    return EmbeddingsResponse(data=data)


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(app, host="0.0.0.0", port=19324, workers=1, log_config=log_config)
