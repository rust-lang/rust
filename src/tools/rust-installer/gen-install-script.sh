#!/bin/bash

set -ue

# Prints the absolute path of a directory to stdout
abs_path() {
    local path="$1"
    # Unset CDPATH because it causes havok: it makes the destination unpredictable
    # and triggers 'cd' to print the path to stdout. Route `cd`'s output to /dev/null
    # for good measure.
    (unset CDPATH && cd "$path" > /dev/null && pwd)
}

src_dir="$(abs_path $(dirname "$0"))"
cargo run --manifest-path="$src_dir/Cargo.toml" -- script "$@"
