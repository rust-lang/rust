#!/bin/sh
# Download sources to build C versions of intrinsics. Once being run,
# `RUST_COMPILER_RT_ROOT` must be set.

set -eux

rust_llvm_version=20.1-2025-02-13

curl -L -o code.tar.gz "https://github.com/rust-lang/llvm-project/archive/rustc/${rust_llvm_version}.tar.gz"
tar xzf code.tar.gz --strip-components 1 llvm-project-rustc-${rust_llvm_version}/compiler-rt
