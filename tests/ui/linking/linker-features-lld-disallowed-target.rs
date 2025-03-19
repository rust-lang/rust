// Check that -CLinker-features=[+-]lld can only be used on x64.
//
//@ check-fail
//@ compile-flags: --target=x86_64-unknown-linux-musl -C linker-features=-lld --crate-type=rlib
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]
