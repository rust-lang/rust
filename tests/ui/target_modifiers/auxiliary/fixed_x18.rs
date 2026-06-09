//@ no-prefer-dynamic
//@ compile-flags: --target aarch64-unknown-none -Zfixed-x18
//@ needs-llvm-components: aarch64

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
