//@ only-aarch64
//@ build-pass
//@ needs-asm-support

#![crate_type = "rlib"]

// AArch64 test corresponding to arm64ec-sve.rs.

use std::arch::asm;

fn f(x: f64) {
    unsafe {
        asm!("", out("p0") _);
        asm!("", out("ffr") _);
    }
}
