#!/bin/sh

set -ex

cargo test --target $TARGET
cargo test --release --target $TARGET
