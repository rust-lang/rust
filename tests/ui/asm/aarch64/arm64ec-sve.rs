//@ add-core-stubs
//@ compile-flags: --target arm64ec-pc-windows-msvc
//@ needs-asm-support
//@ needs-llvm-components: aarch64

#![crate_type = "rlib"]
#![feature(no_core)]
#![no_core]

// SVE cannot be used for Arm64EC
// https://github.com/rust-lang/rust/pull/131332#issuecomment-2401189142

extern crate minicore;
use minicore::*;

fn f(x: f64) {
    unsafe {
        asm!("", out("p0") _);
        //~^ ERROR cannot use register `p0`
        asm!("", out("ffr") _);
        //~^ ERROR cannot use register `ffr`
    }
}
