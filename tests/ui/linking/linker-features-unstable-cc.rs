// Check that only `-C linker-features=-lld` is stable on x64 linux, and that other linker
// features require using `-Z unstable-options`.
//
// Note that, currently, only `lld` is parsed on the CLI, but that other linker features can exist
// internally (`cc`).
//
//@ compile-flags: --target=x86_64-unknown-linux-gnu -C linker-features=+cc --crate-type=rlib
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]

//~? ERROR incorrect value `+cc` for codegen option `linker-features`
