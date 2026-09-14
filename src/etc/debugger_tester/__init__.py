import os

debugger = os.environ.get("DEBUGGER_TESTER_DEBUGGER")

if debugger == "lldb":
    from .lldb.batchmode import main as main
elif debugger == "gdb":
    from .gdb.gdb_commands import ReprCommand as ReprCommand
