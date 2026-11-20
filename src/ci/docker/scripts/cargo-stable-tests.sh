#!/bin/bash

# Run Cargo tests using the stable Rust compiler

TOOLCHAIN=stable

rustup install --no-self-update "${TOOLCHAIN}"

RUSTC=$(rustup +"${TOOLCHAIN}" which rustc)
RUSTDOC=$(rustup +"${TOOLCHAIN}" which rustdoc)

# Run Cargo tests with the given toolchain
../x test cargo --set build.rustc="${RUSTC}" --set build.rustdoc="${RUSTDOC}"
