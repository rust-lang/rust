import os

debugger = os.environ.get("BATCHMODE_DEBUGGER")

if debugger == "lldb":
    from .runner import main as main
elif debugger == "gdb":
    from .gdb_commands import ReprCommand as ReprCommand
