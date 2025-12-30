import dataclasses
import os

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def main():
    server_args = ServerArgs(
        model_path=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        # attention_backend="",
        disable_cuda_graph=True,
    )

    prompts = [
        "List the first ten prime numbers:",
        "The capital of France is",
        "Once upon a time in a land far, far away,",
        "List 10 numbers only contains digit 1:",
    ]
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 32}

    llm = sgl.Engine(**dataclasses.asdict(server_args))

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()
