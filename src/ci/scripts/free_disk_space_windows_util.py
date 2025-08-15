"""
Utilities for Windows disk space cleanup scripts.
"""

import os
from pathlib import Path
import sys


def get_temp_dir() -> Path:
    """Get the temporary directory set by GitHub Actions."""
    return Path(os.environ.get("RUNNER_TEMP"))


def get_pid_file() -> Path:
    return get_temp_dir() / "free-disk-space.pid"


def get_log_file() -> Path:
    return get_temp_dir() / "free-disk-space.log"


def run_main(main_fn):
    exit_code = 1
    try:
        exit_code = main_fn()
    except Exception as e:
        print(f"::error::{e}")
    sys.exit(exit_code)
