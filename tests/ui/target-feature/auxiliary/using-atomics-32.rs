//@ compile-flags: --crate-type=rlib --target=armv7-unknown-linux-gnueabihf --emit=metadata
//@ needs-llvm-components: arm
//@ compile-flags: -Ctarget-feature=+atomics-32
#![feature(no_core)]
#![no_core]
