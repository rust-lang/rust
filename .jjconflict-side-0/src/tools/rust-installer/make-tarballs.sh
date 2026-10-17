#!/bin/sh

set -ue

# Prints the absolute path of a directory to stdout
abs_path() {
    local path="$1"
    # Unset CDPATH because it causes havok: it makes the destination unpredictable
    # and triggers 'cd' to print the path to stdout. Route `cd`'s output to /dev/null
    # for good measure.
    (unset CDPATH && cd "$path" > /dev/null && pwd)
}

# Running cargo will read the libstd Cargo.toml
# which uses the unstable `public-dependency` feature.
export RUSTC_BOOTSTRAP=1

src_dir="$(abs_path $(dirname "$0"))"
$CARGO run --manifest-path="$src_dir/Cargo.toml" -- tarball "$@"
