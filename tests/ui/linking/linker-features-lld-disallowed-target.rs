// Check that -CLinker-features=[+-]lld can only be used on x64.
//
//@ check-fail
//@ compile-flags: --target=x86_64-unknown-linux-musl -C linker-features=-lld --crate-type=rlib
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]

//~? ERROR `-C linker-features` with lld are unstable for the `x86_64-unknown-linux-musl` target, the `-Z unstable-options` flag must also be passed to use it on this target
