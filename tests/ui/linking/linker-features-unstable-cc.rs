// Check that -CLinker-features with anything else than lld requires -Zunstable-options.
//
//@ check-fail
//@ compile-flags: --target=x86_64-unknown-linux-gnu -C linker-features=+cc --crate-type=rlib
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]

//~? ERROR incorrect value `+cc` for codegen option `linker-features` - a list of enabled (`+` prefix) and disabled (`-` prefix) features: `lld` was expected
