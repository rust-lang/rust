#!/bin/bash

# Make sure you double check the diffs after running this script - with great
# power comes great responsibility.
# We deliberately avoid reformatting files with rustfmt comment directives.

cargo build --release

target/release/rustfmt src/lib.rs
target/release/rustfmt src/bin/main.rs
target/release/rustfmt src/cargo-fmt/main.rs

for filename in tests/target/*.rs; do
    if ! grep -q "rustfmt-" "$filename"; then
        target/release/rustfmt $filename
    fi
done
