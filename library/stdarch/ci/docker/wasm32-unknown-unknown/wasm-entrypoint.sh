#!/bin/bash

set -e

# Download an appropriate version of wasm-bindgen based off of what's being used
# in the lock file. Ideally we'd use `wasm-pack` at some point for this!
version=$(grep -A 1 'name = "wasm-bindgen"' Cargo.lock | grep version)
version=$(echo $version | awk '{print $3}' | sed 's/"//g')
curl -L https://github.com/rustwasm/wasm-bindgen/releases/download/$version/wasm-bindgen-$version-x86_64-unknown-linux-musl.tar.gz \
   | tar xzf - -C target
export PATH=$PATH:`pwd`/target/wasm-bindgen-$version-x86_64-unknown-linux-musl
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner
export NODE_ARGS=--experimental-wasm-simd

exec "$@"
