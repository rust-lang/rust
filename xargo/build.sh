#!/bin/sh
cd "$(dirname "$0")"
sed 's/gcc = "0\.3\.50"/gcc = "=0\.3\.50"/' -i ~/.rustup/toolchains/*/lib/rustlib/src/rust/src/libstd/Cargo.toml
RUSTFLAGS='-Zalways-encode-mir -Zmir-emit-validate=1' xargo build
