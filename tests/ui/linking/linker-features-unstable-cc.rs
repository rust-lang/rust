// Check that -CLinker-features with anything else than lld requires -Zunstable-options.
//
//@ check-fail
//@ compile-flags: --target=x86_64-unknown-linux-gnu -C linker-features=+cc --crate-type=rlib
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]
