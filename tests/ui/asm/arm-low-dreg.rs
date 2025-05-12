//@ add-core-stubs
//@ build-pass
//@ compile-flags: --target=armv7-unknown-linux-gnueabihf
//@ needs-llvm-components: arm
#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

// We accidentally classified "d0"..="d15" as dregs, even though they are in dreg_low16,
// and thus didn't compile them on platforms with only 16 dregs.
// Highlighted in https://github.com/rust-lang/rust/issues/126797

extern crate minicore;
use minicore::*;

fn f(x: f64) -> f64 {
    let out: f64;
    unsafe {
        asm!("vmov.f64 d1, d0", out("d1") out, in("d0") x);
    }
    out
}
