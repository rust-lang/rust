"""
Wait for the background Windows disk cleanup process.
"""

import ctypes
import time
from free_disk_space_windows_util import get_pid_file, get_log_file, run_main


def is_process_running(pid: int) -> bool:
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    processHandle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, 0, pid
    )
    if processHandle == 0:
        # The process is not running.
        # If you don't have the sufficient rights to check if a process is running,
        # zero is also returned. But in GitHub Actions we have these rights.
        return False
    else:
        ctypes.windll.kernel32.CloseHandle(processHandle)
        return True


def print_logs():
    """Print the logs from the cleanup script."""
    log_file = get_log_file()
    if log_file.exists():
        print("free-disk-space logs:")
        # Print entire log; replace undecodable bytes to avoid exceptions.
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                print(f.read())
        except Exception as e:
            raise Exception(f"Failed to read log file '{log_file}'") from e
    else:
        print(f"::warning::Log file '{log_file}' not found")


def read_pid_from_file() -> int:
    """Read the PID from the pid file."""

    pid_file = get_pid_file()
    if not pid_file.exists():
        raise Exception(
            f"No background free-disk-space process to wait for: pid file {pid_file} not found"
        )

    pid_file_content = pid_file.read_text().strip()

    # Delete the file if it exists
    pid_file.unlink(missing_ok=True)

    try:
        # Read the first line and convert to int.
        pid = int(pid_file_content.splitlines()[0])
        return pid
    except Exception as e:
        raise Exception(
            f"Error while parsing the pid file with content '{pid_file_content!r}'"
        ) from e


def main() -> int:
    pid = read_pid_from_file()

    # Poll until process exits
    while is_process_running(pid):
        time.sleep(3)

    print_logs()

    return 0


if __name__ == "__main__":
    run_main(main)
