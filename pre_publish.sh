#!/bin/bash

set -e

cd clippy_lints && cargo fmt && cd ..
cargo fmt
cargo test
./util/update_lints.py

