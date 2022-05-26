#!/bin/bash -e
rustup toolchain uninstall wasix || true
sudo apt install python ninja-build
./x.py build
./x.py build --stage 2
rustup toolchain link wasix $pwd/build/x86_64-unknown-linux-gnu/stage2
rustup default wasix
