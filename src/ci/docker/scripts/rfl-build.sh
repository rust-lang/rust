#!/bin/bash

set -euo pipefail

LINUX_VERSION=c13320499ba0efd93174ef6462ae8a7a2933f6e7

# Build rustc, rustdoc and cargo
../x.py build --stage 1 library rustdoc
../x.py build --stage 0 cargo

# Install rustup so that we can use the built toolchain easily, and also
# install bindgen in an easy way.
curl --proto '=https' --tlsv1.2 -sSf -o rustup.sh https://sh.rustup.rs
sh rustup.sh -y --default-toolchain none

source /cargo/env

BUILD_DIR=$(realpath ./build)
rustup toolchain link local "${BUILD_DIR}"/x86_64-unknown-linux-gnu/stage1
rustup default local

mkdir -p rfl
cd rfl

# Remove existing directory to make local builds easier
rm -rf linux || true

# Download Linux at a specific commit
mkdir -p linux
git -C linux init
git -C linux remote add origin https://github.com/torvalds/linux.git
git -C linux fetch --depth 1 origin ${LINUX_VERSION}
git -C linux checkout FETCH_HEAD

# Install bindgen
"${BUILD_DIR}"/x86_64-unknown-linux-gnu/stage0/bin/cargo install \
  --version $(linux/scripts/min-tool-version.sh bindgen) \
  bindgen-cli

# Configure Rust for Linux
cat <<EOF > linux/kernel/configs/rfl-for-rust-ci.config
# CONFIG_WERROR is not set

CONFIG_RUST=y

CONFIG_SAMPLES=y
CONFIG_SAMPLES_RUST=y

CONFIG_SAMPLE_RUST_MINIMAL=m
CONFIG_SAMPLE_RUST_PRINT=y

CONFIG_RUST_PHYLIB_ABSTRACTIONS=y
CONFIG_AX88796B_PHY=y
CONFIG_AX88796B_RUST_PHY=y

CONFIG_KUNIT=y
CONFIG_RUST_KERNEL_DOCTESTS=y
EOF

make -C linux LLVM=1 -j$(($(nproc) + 1)) \
    rustavailable \
    defconfig \
    rfl-for-rust-ci.config

make -C linux LLVM=1 -j$(($(nproc) + 1)) \
    samples/rust/rust_minimal.o \
    samples/rust/rust_print.o \
    drivers/net/phy/ax88796b_rust.o
