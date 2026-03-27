"""Entry point for the BCI Signal Processor + State Server."""

import argparse
import logging
import signal
import sys

import uvicorn

from . import config
from .server import configure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BCI Signal Processor + State Server"
    )
    parser.add_argument(
        "--port", type=int, default=config.PORT,
        help=f"HTTP server port (default: {config.PORT})",
    )
    parser.add_argument(
        "--host", type=str, default=config.HOST,
        help=f"HTTP server host (default: {config.HOST})",
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="Use BrainFlow synthetic board (default: True)",
    )
    parser.add_argument(
        "--no-synthetic", action="store_false", dest="synthetic",
        help="Use real hardware (Galea) instead of synthetic board",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Configure server before uvicorn starts the app
    configure(synthetic=args.synthetic)

    # Graceful shutdown on SIGINT/SIGTERM
    def handle_signal(signum, frame):
        logging.getLogger(__name__).info("Received signal %d, shutting down...", signum)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    uvicorn.run(
        "src.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
