#!/bin/bash

# Make sure you double check the diffs after running this script - with great
# power comes great responsibility.
# We deliberately avoid reformatting files with rustfmt comment directives.

cargo build --release

target/release/rustfmt --write-mode=overwrite src/lib.rs
target/release/rustfmt --write-mode=overwrite src/bin/rustfmt.rs
target/release/rustfmt --write-mode=overwrite src/bin/cargo-fmt.rs
target/release/rustfmt --write-mode=overwrite tests/system.rs

for filename in tests/target/*.rs; do
    if ! grep -q "rustfmt-" "$filename"; then
        target/release/rustfmt --write-mode=overwrite $filename
    fi
done
