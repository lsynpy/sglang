"""
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500 SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=64 \
MODEL=Qwen/Qwen3-1.7B \
python -m sglang.launch_server \
    --dtype float16 \
    --model-path $MODEL \
    --attention-backend triton \
    --mem-fraction-static 0.5 \
    --decode-log-interval 1 \
    --cuda-graph-bs (seq 1 64) \
    --speculative-algorithm NGRAM \
    --speculative-num-draft-tokens 6 \
    --speculative-ngram-min-match-window-size 1 \
    --speculative-ngram-max-match-window-size 12 \
    --speculative-ngram-min-bfs-breadth 1 \
    --speculative-ngram-max-bfs-breadth 10 \
    --speculative-ngram-branch-length 18 \
    --speculative-ngram-capacity 10000000

python -m sglang.test.send_one
"""

import os

import sglang as sgl
import torch

os.environ["SGLANG_LOG_LEVEL"] = "DEBUG"

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


PROMPTS = [
    "List the first ten prime numbers:",
    "The capital of France is",
    "Once upon a time in a land far, far away,",
    "List 10 numbers only contains digit 1:",
    "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:",
]

TARGET = os.path.expanduser("~/huggingface/Qwen3-1.7B")

NUM_DRAFT_TOKENS = 3
TEMPERATURE = 0
OUTPUT_LEN = 8


def main():
    llm = sgl.Engine(
        model_path=TARGET,
        tp_size=1,
        chunked_prefill_size=-1,
        mem_fraction_static=0.5,
        speculative_algorithm="NGRAM",
        speculative_num_draft_tokens=NUM_DRAFT_TOKENS,
        speculative_ngram_min_match_window_size=1,
        speculative_ngram_max_match_window_size=12,
        speculative_ngram_min_bfs_breadth=1,
        speculative_ngram_max_bfs_breadth=10,
        speculative_ngram_branch_length=18,
        speculative_ngram_capacity=10000000,
        max_total_tokens=32,
        max_running_requests=4,
    )

    sampling_params = {"temperature": TEMPERATURE, "max_new_tokens": OUTPUT_LEN}

    outputs = llm.generate(PROMPTS, sampling_params)

    for prompt, output in zip(PROMPTS, outputs):
        generated_text = output["text"]
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    total_num_output_tokens = sum(output["meta_info"]["completion_tokens"] for output in outputs)

    if outputs:
        first_output_meta = outputs[0]["meta_info"]
        avg_spec_accept_length = first_output_meta.get("spec_accept_length", 0)
        num_accepted_tokens = first_output_meta.get("spec_accept_token_num", 0)
        num_draft_tokens = first_output_meta.get("spec_draft_token_num", 0)
        num_verify_calls = first_output_meta.get("spec_verify_ct", 0)

        num_drafts = num_verify_calls

        acceptance_length = (
            avg_spec_accept_length
            if avg_spec_accept_length > 0
            else (1 + (num_accepted_tokens / num_drafts if num_drafts > 0 else 0))
        )
    else:
        num_drafts = 0
        num_draft_tokens = 0
        num_accepted_tokens = 0
        acceptance_length = 0

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    acceptance_counts = [0] * NUM_DRAFT_TOKENS
    for i in range(len(acceptance_counts)):
        print(f"acceptance at token {i}: 0.00")


if __name__ == "__main__":
    main()