# LLDB Python script to handle Cargo `<exe>.trim-paths.jsonl` files from the trim-paths feature
#  - https://github.com/rust-lang/cargo/issues/12137
#  - https://github.com/rust-lang/rust/issues/111540

import json
import os
import sys
import lldb
import threading


def _process_v1_trim_paths(debugger, lines, trim_paths_path):
    for idx, line in enumerate(lines[2:], start=3):
        try:
            entry = json.loads(line)
            if "from" in entry and "to" in entry:
                # LLDB syntax: settings append target.source-map <from> <to>
                cmd = f'settings append target.source-map "{entry["from"]}" "{entry["to"]}"'
                debugger.HandleCommand(cmd)
        except json.JSONDecodeError:
            print(
                f"(rust-lldb) warning: invalid JSON on line {idx} of {trim_paths_path}",
                file=sys.stderr,
            )


def _load_trim_paths(debugger, filepath):
    trim_paths_path = f"{filepath}.trim-paths.jsonl"

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
            ver = header.get("v")
        except json.JSONDecodeError:
            print(
                f"(rust-lldb) warning: header line 1 of {trim_paths_path} is not valid JSON",
                file=sys.stderr,
            )
            return

        # We only handle version 1
        if ver == 1:
            _process_v1_trim_paths(debugger, lines, trim_paths_path)
        else:
            print(
                f"(rust-lldb) warning: unsupported trim-paths version {ver}: {trim_paths_path}",
                file=sys.stderr,
            )
    except Exception as e:
        print(
            f"(rust-lldb) warning: failed to process trim-paths mappings: {e} of {trim_paths_path}",
            file=sys.stderr,
        )


def _on_target(debugger, target):
    # https://lldb.llvm.org/python_reference/lldb.SBTarget-class.html
    if not target or not target.IsValid():
        return

    for module in target.module_iter():
        # SBFileSpec for the module's file on the host machine
        # https://lldb.llvm.org/python_reference/lldb.SBFileSpec-class.html
        file_spec = module.GetFileSpec()

        filepath = os.path.join(file_spec.GetDirectory(), file_spec.GetFilename())

        _load_trim_paths(debugger, filepath)


def _listen_for_new_targets(debugger):
    listener = lldb.SBListener("rust_lldb_target_listener")

    # Listen for loaded modules
    listener.StartListeningForEventClass(
        debugger,
        lldb.SBTarget.GetBroadcasterClassName(),
        lldb.SBTarget.eBroadcastBitModulesLoaded,
    )

    while True:
        event = lldb.SBEvent()

        if not listener.WaitForEvent(1, event):
            continue
        if not lldb.SBTarget.EventIsTargetEvent(event):
            continue

        target = debugger.GetSelectedTarget()
        _on_target(debugger, target)


def __lldb_init_module(debugger, internal_dict):
    # Process any existing target present on module load
    target = debugger.GetSelectedTarget()
    _on_target(debugger, target)

    # Start background thread to listen to event in the background
    listener_thread = threading.Thread(
        target=_listen_for_new_targets, args=(debugger,), daemon=True
    )
    listener_thread.start()
