#!/bin/sh

# Exit if anything fails
set -e

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "-help" ] || [ "$1" = "--help" ]; then
    echo "
rust-gdbgui
===========
gdbgui - https://gdbgui.com - is a graphical front-end to GDB
that runs in a browser. This script invokes gdbgui with the Rust
pretty printers loaded.

Simple usage  : rust-gdbgui target/debug/myprog
With arguments: rust-gdbgui 'target/debug/myprog arg1 arg2...'
  (note the quotes)


Hints
=====
gdbgui won't be able to find the rust 'main' method automatically, so
in its options make sure to disable the 'Add breakpoint to main after
loading executable' setting to avoid a 'File not found: main' warning
on startup.

Instead, type 'main' into gdbgui's file browser and you should get
auto-completion on the filename. Just pick 'main.rs', add a breakpoint
by clicking in the line number gutter, and type 'r' or hit the Restart
icon to start your program running.
"
    exit 0
fi

# Prefer rustc in the same directory as this script
DIR="$(dirname "$0")"
if [ -x "$DIR/rustc" ]; then
  RUSTC="$DIR/rustc"
else
  RUSTC="rustc"
fi

# Find out where the pretty printer Python module is
RUSTC_SYSROOT="$("$RUSTC" --print=sysroot)"
GDB_PYTHON_MODULE_DIRECTORY="$RUSTC_SYSROOT/lib/rustlib/etc"
# Get the commit hash for path remapping
RUSTC_COMMIT_HASH="$("$RUSTC" -vV | sed -n 's/commit-hash: \([a-zA-Z0-9_]*\)/\1/p')"

# Set the environment variable `RUST_GDB` to overwrite the call to a
# different/specific command (defaults to `gdb`).
RUST_GDB="${RUST_GDB:-gdb}"

# Set the environment variable `RUST_GDBGUI` to overwrite the call to a
# different/specific command (defaults to `gdbgui`).
RUST_GDBGUI="${RUST_GDBGUI:-gdbgui}"

# These arguments get passed through to GDB and make it load the
# Rust pretty printers.
GDB_ARGS="--directory=\"$GDB_PYTHON_MODULE_DIRECTORY\" \
   -iex \"add-auto-load-safe-path $GDB_PYTHON_MODULE_DIRECTORY\" \
   -iex \"set substitute-path /rustc/$RUSTC_COMMIT_HASH $RUSTC_SYSROOT/lib/rustlib/src/rust\""

# Finally we execute gdbgui.
PYTHONPATH="$PYTHONPATH:$GDB_PYTHON_MODULE_DIRECTORY" \
  exec ${RUST_GDBGUI} \
  --gdb-cmd "${RUST_GDB} ${GDB_ARGS}" \
  "${@}"

