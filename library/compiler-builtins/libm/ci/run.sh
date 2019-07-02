#!/usr/bin/env sh

set -ex
TARGET=$1

CMD="cargo test --all --no-default-features --target $TARGET"

$CMD
$CMD --release

$CMD --features 'stable'
$CMD --release --features 'stable'

$CMD --features 'stable checked musl-reference-tests'
$CMD --release --features  'stable checked musl-reference-tests'
