#!/usr/bin/env bash

# Builds documentation for all target triples that we have a registered URL for
# in liblibc. This scrapes the list of triples to document from `src/lib.rs`
# which has a bunch of `html_root_url` directives we pick up.

set -e

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

  cargo build --verbose --target "${target}" --manifest-path crates/stdsimd/Cargo.toml

  rustdoc --verbose --target "${target}" \
          -o "target/doc/${arch}" crates/coresimd/src/lib.rs \
          --crate-name coresimd \
          --library-path "target/${target}/debug/deps"
  rustdoc --verbose --target "${target}" \
          -o "target/doc/${arch}" crates/stdsimd/src/lib.rs \
          --crate-name stdsimd \
          --library-path "target/${target}/debug/deps" \
          --extern cfg_if="$(ls target/"${target}"/debug/deps/libcfg_if-*.rlib)" \
          --extern libc="$(ls target/"${target}"/debug/deps/liblibc-*.rlib)"
}

dox i686 i686-unknown-linux-gnu
dox x86_64 x86_64-unknown-linux-gnu
dox arm armv7-unknown-linux-gnueabihf
dox aarch64 aarch64-unknown-linux-gnu
# dox powerpc powerpc-unknown-linux-gnu
dox powerpc64le powerpc64le-unknown-linux-gnu
dox mips mips-unknown-linux-gnu
dox mips64 mips64-unknown-linux-gnuabi64
dox wasm32 wasm32-unknown-unknown

# If we're on travis, not a PR, and on the right branch, publish!
if [ "$TRAVIS_PULL_REQUEST" = "false" ] && [ "$TRAVIS_BRANCH" = "master" ]; then
  pip install ghp_import --install-option="--prefix=$HOME/.local"
  "${HOME}/.local/bin/ghp-import" -n target/doc
  git push -qf "https://${GH_TOKEN}@github.com/${TRAVIS_REPO_SLUG}.git" gh-pages
fi
