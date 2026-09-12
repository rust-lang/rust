//! A crate that uses the default codegen backend and an LLVM target feature.

//@ compile-flags: --target arm-unknown-linux-gnueabi --emit=metadata
//@ needs-llvm-components: arm
//@ compile-flags: -Zllvm-target-feature=+thumb2

#![feature(no_core)]
#![no_core]
