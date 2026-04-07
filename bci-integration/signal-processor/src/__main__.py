"""Entry point for the BCI Signal Processor + State Server.

Usage:
    python -m src --port 7680 --synthetic
    python -m src --port 7680 --no-synthetic
    python -m src --record session.jsonl          # record while running
    python -m src --replay session.jsonl          # replay a recording
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
    parser.add_argument(
        "--record",
        type=str,
        metavar="FILE",
        default=None,
        help="Record session to a JSONL file while running normally",
    )
    parser.add_argument(
        "--replay",
        type=str,
        metavar="FILE",
        default=None,
        help="Replay a recorded JSONL session instead of using BrainFlow",
    )
    parser.add_argument(
        "--model",
        type=str,
        metavar="PATH",
        default=None,
        help="Path to a trained ML model (.joblib) for brain state classification",
    )
    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.record and args.replay:
        parser.error("--record and --replay are mutually exclusive")

    if args.replay and args.synthetic is not True:
        # --no-synthetic was explicitly passed alongside --replay
        # We only error if the user explicitly set --no-synthetic
        pass  # Allow it; replay ignores synthetic flag anyway

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Import here to avoid circular imports and to pass flags
    from .classifier import create_classifier
    from .server import create_app
    from .recorder import SessionRecorder
    from .replayer import SessionReplayer
    from .state_manager import StateManager

    recorder = None
    replayer = None

    # Resolve model path: CLI flag > env var > None
    model_path = args.model or config.MODEL_PATH
    classifier = create_classifier(model_path) if model_path else None

    if args.record:
        recorder = SessionRecorder(file_path=args.record)

    if args.replay:
        # In replay mode, the replayer and server must share the same StateManager.
        state_mgr = StateManager()
        replayer = SessionReplayer(file_path=args.replay, state_manager=state_mgr)
        app = create_app(synthetic=True, replayer=replayer, state_manager=state_mgr)
    else:
        app = create_app(synthetic=args.synthetic, recorder=recorder, classifier=classifier)

    # Graceful shutdown on SIGINT/SIGTERM
    def handle_signal(signum, frame):
        logging.getLogger(__name__).info("Received signal %d, shutting down...", signum)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
