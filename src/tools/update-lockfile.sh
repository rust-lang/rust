#!/bin/bash

# Updates the workspaces in `.`, `library` and `src/tools/rustbook`
# Logs are written to `cargo_update.log`
# Used as part of regular dependency bumps

set -euo pipefail

printf "\ncompiler & tools dependencies:" > cargo_update.log
# Remove first line that always just says "Updating crates.io index"
cargo update 2>&1 | sed '/crates.io index/d' | \
    tee -a cargo_update.log
printf "\nlibrary dependencies:" >> cargo_update.log
cargo update --manifest-path library/Cargo.toml 2>&1 | sed '/crates.io index/d' | \
    tee -a cargo_update.log
printf "\nrustbook dependencies:" >> cargo_update.log
cargo update --manifest-path src/tools/rustbook/Cargo.toml 2>&1 | sed '/crates.io index/d' | \
    tee -a cargo_update.log
