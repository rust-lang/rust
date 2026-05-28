//@ build-pass
//@ add-minicore
//@ compile-flags: --target aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc
#![crate_type = "rlib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

// AArch64 test corresponding to arm64ec-sve.rs.

fn f(x: f64) {
    unsafe {
        asm!("", out("p0") _);
        asm!("", out("ffr") _);
    }
}
