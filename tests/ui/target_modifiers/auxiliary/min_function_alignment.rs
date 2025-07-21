//@ no-prefer-dynamic
//@ compile-flags: --target aarch64-unknown-none -Zmin-function-alignment=32
//@ needs-llvm-components: aarch64

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
