#!/usr/bin/env bash

# Builds documentation for all target triples that we have a registered URL for
# in liblibc. This scrapes the list of triples to document from `src/lib.rs`
# which has a bunch of `html_root_url` directives we pick up.

set -ex

dox() {
  if [ "$CI" != "" ]; then
    rustup target add "${1}" || true
  fi

  cargo clean --target "${1}"

  cargo build --verbose --target "${1}" --manifest-path crates/core_arch/Cargo.toml
  cargo build --verbose --target "${1}" --manifest-path crates/std_detect/Cargo.toml

  cargo doc --verbose --target "${1}" --manifest-path crates/core_arch/Cargo.toml
  cargo doc --verbose --target "${1}" --manifest-path crates/std_detect/Cargo.toml
}

if [ -z "$1" ]; then
  dox i686-unknown-linux-gnu
  dox x86_64-unknown-linux-gnu
  # Disabled temporarily,
  # See https://github.com/rust-lang/rust/issues/134511
  #dox armv7-unknown-linux-gnueabihf
  dox aarch64-unknown-linux-gnu
  dox powerpc-unknown-linux-gnu
  dox powerpc64le-unknown-linux-gnu
  dox loongarch64-unknown-linux-gnu
  # MIPS targets disabled since they are dropped to tier 3.
  # See https://github.com/rust-lang/compiler-team/issues/648
  #dox mips-unknown-linux-gnu
  #dox mips64-unknown-linux-gnuabi64
  dox wasm32-unknown-unknown
  dox nvptx64-nvidia-cuda
else
  dox "${1}"
fi