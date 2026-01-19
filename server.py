from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs


def main():
    server_args = ServerArgs(
        model_path="Qwen/Qwen3-0.6B",
        # enable_trace=True,  # tracing
        # otlp_traces_endpoint="0.0.0.0:4317",  # tracing
    )

    launch_server(server_args)


if __name__ == "__main__":
    main()
