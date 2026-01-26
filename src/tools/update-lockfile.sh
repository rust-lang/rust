#!/bin/sh

# Updates the workspaces in `.`, `library` and `src/tools/rustbook`
# Logs are written to `cargo_update.log`
# Used as part of regular dependency bumps

set -euo pipefail

echo -e "\ncompiler & tools dependencies:" > cargo_update.log
# Remove first line that always just says "Updating crates.io index"
cargo update 2>&1 | sed '/crates.io index/d' | \
    tee -a cargo_update.log
echo -e "\nlibrary dependencies:" >> cargo_update.log
cargo update --manifest-path library/Cargo.toml 2>&1 | sed '/crates.io index/d' | \
    tee -a cargo_update.log
echo -e "\nrustbook dependencies:" >> cargo_update.log
cargo update --manifest-path src/tools/rustbook/Cargo.toml 2>&1 | sed '/crates.io index/d' | \
    tee -a cargo_update.log
