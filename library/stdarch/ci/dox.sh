#!/usr/bin/env bash

# Builds documentation for all target triples that we have a registered URL for
# in liblibc. This scrapes the list of triples to document from `src/lib.rs`
# which has a bunch of `html_root_url` directives we pick up.

set -ex

rm -rf target/doc
mkdir -p target/doc

dox() {
  local arch=$1
  local target=$2

  echo "documenting ${arch}"

  if [ "$CI" != "" ]; then
    rustup target add "${target}" || true
  fi

  rm -rf "target/doc/${arch}"
  mkdir "target/doc/${arch}"

  cargo build --verbose --target "${target}" --manifest-path crates/core_arch/Cargo.toml
  cargo build --verbose --target "${target}" --manifest-path crates/std_detect/Cargo.toml

  rustdoc --verbose --target "${target}" \
          -o "target/doc/${arch}" crates/core_arch/src/lib.rs \
          --edition=2018 \
          --crate-name core_arch \
          --library-path "target/${target}/debug/deps"
  rustdoc --verbose --target "${target}" \
          -o "target/doc/${arch}" crates/std_detect/src/lib.rs \
          --edition=2018 \
          --crate-name std_detect \
          --library-path "target/${target}/debug/deps" \
          --extern cfg_if="$(ls target/"${target}"/debug/deps/libcfg_if-*.rlib)" \
          --extern libc="$(ls target/"${target}"/debug/deps/liblibc-*.rlib)"
}

dox i686 i686-unknown-linux-gnu
dox x86_64 x86_64-unknown-linux-gnu
dox arm armv7-unknown-linux-gnueabihf
dox aarch64 aarch64-unknown-linux-gnu
dox powerpc powerpc-unknown-linux-gnu
dox powerpc64le powerpc64le-unknown-linux-gnu
# MIPS targets disabled since they are dropped to tier 3.
# See https://github.com/rust-lang/compiler-team/issues/648
#dox mips mips-unknown-linux-gnu
#dox mips64 mips64-unknown-linux-gnuabi64
dox wasm32 wasm32-unknown-unknown
