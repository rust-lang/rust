// Check that non-lld linker features require using `-Z unstable-options`.
//
// Note that, currently, only `lld` is parsed on the CLI, but that other linker features can exist
// internally (`cc`).
//
//@ compile-flags: --target=x86_64-unknown-linux-gnu -C linker-features=+cc --crate-type=rlib
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]

//~? ERROR incorrect value `+cc` for codegen option `linker-features`
