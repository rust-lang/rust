#!/bin/sh

set -eux

target="$1"

cmd="cargo test --all --target $target"

# Needed for no-panic to correct detect a lack of panics
export RUSTFLAGS="$RUSTFLAGS -Ccodegen-units=1"

# stable by default
$cmd
$cmd --release

# unstable with a feature
$cmd --features 'unstable'
$cmd --release --features 'unstable'

# also run the reference tests
$cmd --features 'unstable libm-test/musl-bitwise-tests'
$cmd --release --features 'unstable libm-test/musl-bitwise-tests'
