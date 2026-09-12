#!/usr/bin/env bash

set -ex

# Build the components of Rust that are needed by the Fuchsia build system.
python3 ../x.py install \
    --target=$TARGETS \
    compiler/rustc library/std clippy rustfmt src

# The Fuchsia build system requires that shared libraries that are used inside Fuchsia are stripped,
# so use `llvm-objcopy` to remove the debug symbols and strip the libraries, but keeping the
# `.rustc` section needed by rust for dynamic linking.
find /checkout/obj/install/lib/rustlib/*-fuchsia/lib \
    -type f \
    -name "*.so" \
    -exec sh \
    -c '/usr/local/bin/llvm-objcopy --only-keep-debug "$1" "$1.debug" && \
        /usr/local/bin/llvm-objcopy --strip-all --keep-section=.rustc "$1"' \
    _ \
    {} \
    \;
