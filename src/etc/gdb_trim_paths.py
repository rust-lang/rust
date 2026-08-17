# GDB Python script to handle Cargo `<exe>.trim-paths.jsonl` files from the trim-paths feature
#  - https://github.com/rust-lang/cargo/issues/12137
#  - https://github.com/rust-lang/rust/issues/111540

import json
import os
import gdb
import sys


# https://doc.guix.gnu.org/gdb/16.3/en/html_node/Source-Path.html#index-set-substitute_002dpath
def _process_v1_trim_paths(lines, trim_paths_path):
    for idx, line in enumerate(lines[2:], start=3):
        try:
            entry = json.loads(line)
            if "from" in entry and "to" in entry:
                cmd = f'set substitute-path "{entry["from"]}" "{entry["to"]}"'
                gdb.execute(cmd)
        except json.JSONDecodeError:
            print(
                f"(rust-gdb) warning: invalid JSON on line {idx} of {trim_paths_path}",
                file=sys.stderr,
            )


def _load_trim_paths(filepath):
    trim_paths_path = f"{filepath}.trim-paths.jsonl"

    # FIXME: It might be worth looking into the debuginfod fetch content if the local file
    # doesn't exists (maybe with `debuginfod-find debuginfo`).
    if not os.path.isfile(trim_paths_path):
        return

    try:
        # Load all the lines of the trim-paths file
        with open(trim_paths_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f]

        # Abort if we have less than 3 lines as that means that we cannot have any
        # substitutions (header + metadata is already 2 lines)
        if not lines or len(lines) < 3:
            return

        # Try loading the header line, which contains the version (v) field
        try:
            header = json.loads(lines[0])
            ver = header["v"]
        except json.JSONDecodeError:
            print(
                f"(rust-gdb) warning: header line 1 of {trim_paths_path} is not valid JSON",
                file=sys.stderr,
            )
            return

        # We only handle version 1
        if ver == 1:
            _process_v1_trim_paths(lines, trim_paths_path)
        else:
            print(
                f"(rust-gdb) warning: unsupported trim-paths version {ver}: {trim_paths_path}",
                file=sys.stderr,
            )
    except Exception as e:
        print(
            f"(rust-gdb) warning: failed to process trim-paths mappings: {e} of {trim_paths_path}",
            file=sys.stderr,
        )


def _on_objfile(objfile):
    filepath = objfile.filename

    if not filepath or not objfile.is_valid() or not os.path.isfile(filepath):
        return

    _load_trim_paths(filepath)


def _on_progspace(progspace):
    filepath = progspace.filename

    if not filepath or not os.path.isfile(filepath):
        return

    _load_trim_paths(filepath)


def _on_new_objfile(event):
    _on_objfile(event.new_objfile)


def _on_executable_changed(event):
    _on_progspace(event.progspace)


# Setup the events for new objfile (so) and new executable
# https://doc.guix.gnu.org/gdb/16.3/en/html_node/Events-In-Python.html
#
# FIXME: should we handle clear/free events?
gdb.events.new_objfile.connect(_on_new_objfile)
gdb.events.executable_changed.connect(_on_executable_changed)

# Load the trim-paths files for the already loaded objfiles
for objfile in gdb.objfiles():
    _on_objfile(objfile)
