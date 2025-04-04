// Check that `-C linker-features=[+-]lld` is only stable on x64 linux, and needs `-Z
// unstable-options` elsewhere.

// ignore-tidy-linelength

//@ revisions: positive negative
//@ [negative] compile-flags: --target=x86_64-unknown-linux-musl -C linker-features=-lld --crate-type=rlib
//@ [negative] needs-llvm-components: x86
//@ [positive] compile-flags: --target=x86_64-unknown-linux-musl -C linker-features=+lld --crate-type=rlib
//@ [positive] needs-llvm-components: x86

#![feature(no_core)]
#![no_core]

//[negative]~? ERROR `-C linker-features=-lld` is unstable on the `x86_64-unknown-linux-musl` target. The `-Z unstable-options` flag must also be passed to use it on this target
//[positive]~? ERROR `-C linker-features=+lld` is unstable on the `x86_64-unknown-linux-musl` target. The `-Z unstable-options` flag must also be passed to use it on this target
