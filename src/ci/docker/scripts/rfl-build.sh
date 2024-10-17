#!/bin/bash

set -euo pipefail

LINUX_VERSION=v6.12-rc2

# Build rustc, rustdoc, cargo, clippy-driver and rustfmt
../x.py build --stage 2 library rustdoc clippy rustfmt
../x.py build --stage 0 cargo

# Install rustup so that we can use the built toolchain easily, and also
# install bindgen in an easy way.
curl --proto '=https' --tlsv1.2 -sSf -o rustup.sh https://sh.rustup.rs
sh rustup.sh -y --default-toolchain none

source /cargo/env

BUILD_DIR=$(realpath ./build)
rustup toolchain link local "${BUILD_DIR}"/x86_64-unknown-linux-gnu/stage2
rustup default local

mkdir -p rfl
cd rfl

# Remove existing directory to make local builds easier
rm -rf linux || true

# Download Linux at a specific commit
mkdir -p linux
git -C linux init
git -C linux remote add origin https://github.com/Rust-for-Linux/linux.git
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

BUILD_TARGETS="
    samples/rust/rust_minimal.o
    samples/rust/rust_print.o
    drivers/net/phy/ax88796b_rust.o
    rust/doctests_kernel_generated.o
"

# Build a few Rust targets
#
# This does not include building the C side of the kernel nor linking,
# which can find other issues, but it is much faster.
#
# This includes transforming `rustdoc` tests into KUnit ones thanks to
# `CONFIG_RUST_KERNEL_DOCTESTS=y` above (which, for the moment, uses the
# unstable `--test-builder` and `--no-run`).
make -C linux LLVM=1 -j$(($(nproc) + 1)) \
    $BUILD_TARGETS

# Generate documentation
make -C linux LLVM=1 -j$(($(nproc) + 1)) \
    rustdoc

# Build macro expanded source (`-Zunpretty=expanded`)
#
# This target also formats the macro expanded code, thus it is also
# intended to catch ICEs with formatting `-Zunpretty=expanded` output
# like https://github.com/rust-lang/rustfmt/issues/6105.
make -C linux LLVM=1 -j$(($(nproc) + 1)) \
    samples/rust/rust_minimal.rsi

# Re-build with Clippy enabled
#
# This should not introduce Clippy errors, since `CONFIG_WERROR` is not
# set (thus no `-Dwarnings`) and the kernel uses `-W` for all Clippy
# lints, including `clippy::all`. However, it could catch ICEs.
make -C linux LLVM=1 -j$(($(nproc) + 1)) CLIPPY=1 \
    $BUILD_TARGETS

# Format the code
#
# This returns successfully even if there were changes, i.e. it is not
# a check.
make -C linux LLVM=1 -j$(($(nproc) + 1)) \
    rustfmt
