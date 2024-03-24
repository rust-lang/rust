#!/bin/bash
set -e

rustup toolchain install nightly --component miri
rustup override set nightly
cargo miri setup

export MIRIFLAGS="-Zmiri-strict-provenance"

cargo miri test
cargo miri test --target mips64-unknown-linux-gnuabi64
