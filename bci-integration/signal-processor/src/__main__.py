"""Entry point for the BCI Signal Processor + State Server.

Usage:
    python -m src --port 7680 --synthetic
    python -m src --port 7680 --no-synthetic
"""

import argparse
import logging
import signal
import sys

import uvicorn

from . import config


def main() -> None:
    parser = argparse.ArgumentParser(description="BCI Signal Processor + State Server")
    parser.add_argument("--port", type=int, default=config.PORT, help=f"Server port (default: {config.PORT})")
    parser.add_argument("--host", type=str, default=config.HOST, help=f"Server host (default: {config.HOST})")
    parser.add_argument(
        "--synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use BrainFlow synthetic board (default: True)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Import here to avoid circular imports and to pass synthetic flag
    from .server import create_app

    app = create_app(synthetic=args.synthetic)

    # Graceful shutdown on SIGINT/SIGTERM
    def handle_signal(signum, frame):
        logging.getLogger(__name__).info("Received signal %d, shutting down...", signum)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
