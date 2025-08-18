"""
Start freeing disk space on Windows in the background by launching
the PowerShell cleanup script, and recording the PID in a file,
so later steps can wait for completion.
"""

import subprocess
from pathlib import Path
from free_disk_space_windows_util import get_pid_file, get_log_file, run_main


def get_cleanup_script() -> Path:
    script_dir = Path(__file__).resolve().parent
    cleanup_script = script_dir / "free-disk-space-windows.ps1"
    if not cleanup_script.exists():
        raise Exception(f"Cleanup script '{cleanup_script}' not found")
    return cleanup_script


def write_pid(pid: int):
    pid_file = get_pid_file()
    if pid_file.exists():
        raise Exception(f"Pid file '{pid_file}' already exists")
    pid_file.write_text(str(pid))
    print(f"wrote pid {pid} in file {pid_file}")


def launch_cleanup_process():
    cleanup_script = get_cleanup_script()
    log_file_path = get_log_file()
    # Launch the PowerShell cleanup in the background and redirect logs.
    try:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [
                    "pwsh",
                    # Suppress PowerShell startup banner/logo for cleaner logs.
                    "-NoLogo",
                    # Don't load user/system profiles. Ensures a clean, predictable environment.
                    "-NoProfile",
                    # Disable interactive prompts. Required for CI to avoid hangs.
                    "-NonInteractive",
                    # Execute the specified script file (next argument).
                    "-File",
                    str(cleanup_script),
                ],
                # Write child stdout to the log file.
                stdout=log_file,
                # Merge stderr into stdout for a single, ordered log stream.
                stderr=subprocess.STDOUT,
            )
            print(
                f"Started free-disk-space cleanup in background. "
                f"pid={proc.pid}; log_file={log_file_path}"
            )
            return proc
    except FileNotFoundError as e:
        raise Exception("pwsh not found on PATH; cannot start disk cleanup.") from e


def main() -> int:
    proc = launch_cleanup_process()

    # Write pid of the process to a file, so that later steps can read it and wait
    # until the process completes.
    write_pid(proc.pid)

    return 0


if __name__ == "__main__":
    run_main(main)
