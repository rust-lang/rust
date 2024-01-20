#!/usr/bin/env bash

set -ex

# Originally from https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2023.10.18/riscv32-elf-ubuntu-22.04-gcc-nightly-2023.10.18-nightly.tar.gz
curl -L https://ci-mirrors.rust-lang.org/rustc/riscv32-elf-ubuntu-22.04-gcc-nightly-2023.10.18-nightly.tar.gz \
| tar --extract --gz --strip 1 --directory /usr/local
