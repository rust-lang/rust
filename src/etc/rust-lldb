#!/bin/sh

# Exit if anything fails
set -e

# Find the host triple so we can find lldb in rustlib.
host=$(rustc -vV | sed -n -e 's/^host: //p')

# Find out where to look for the pretty printer Python module
RUSTC_SYSROOT=$(rustc --print sysroot)
RUST_LLDB="$RUSTC_SYSROOT/lib/rustlib/$host/bin/lldb"

lldb=lldb
if [ -f "$RUST_LLDB" ]; then
    lldb="$RUST_LLDB"
else
    if ! command -v "$lldb" > /dev/null; then
        echo "$lldb not found! Please install it." >&2
        exit 1
    else
        LLDB_VERSION=$("$lldb" --version | cut -d ' ' -f3)

        if [ "$LLDB_VERSION" = "3.5.0" ]; then
            cat << EOF >&2
***
WARNING: This version of LLDB has known issues with Rust and cannot display the contents of local variables!
***
EOF
        fi
    fi
fi

script_import="command script import \"$RUSTC_SYSROOT/lib/rustlib/etc/lldb_lookup.py\""
commands_file="$RUSTC_SYSROOT/lib/rustlib/etc/lldb_commands"

# Call LLDB with the commands added to the argument list
exec "$lldb" --one-line-before-file "$script_import" --source-before-file "$commands_file" "$@"
