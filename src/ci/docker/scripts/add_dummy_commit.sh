#!/bin/bash

set -ex

if [ "$READ_ONLY_SRC" = "0" ]; then
    # `core::builder::tests::ci_rustc_if_unchanged_logic` bootstrap test ensures that
    # "download-rustc=if-unchanged" logic don't use CI rustc while there are changes on
    # compiler and/or library. Here we are adding a dummy commit on compiler and running
    # that test to make sure we never download CI rustc with a change on the compiler tree.
    echo "" >> ../compiler/rustc/src/main.rs
    git config --global user.email "dummy@dummy.com"
    git config --global user.name "dummy"
    git add ../compiler/rustc/src/main.rs
    git commit -m "test commit for rust.download-rustc=if-unchanged logic"
    DISABLE_CI_RUSTC_IF_INCOMPATIBLE=0 ../x.py test bootstrap \
        -- core::builder::tests::ci_rustc_if_unchanged_logic
    # Revert the dummy commit
    git reset --hard HEAD~1
fi
