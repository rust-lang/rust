# Add this folder to the python sys path; GDB Python-interpreter will now find modules in this path
import sys
from os import path

self_dir = path.dirname(path.realpath(__file__))
sys.path.append(self_dir)

# ruff: noqa: E402
import gdb
import gdb_lookup

# current_objfile can be none; even with `gdb foo-app`; sourcing this file after gdb init now works
try:
    gdb_lookup.register_printers(gdb.current_objfile())
except Exception:
    gdb_lookup.register_printers(gdb.selected_inferior().progspace)
