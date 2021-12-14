// only-x86_64
// Make sure rustc doesn't ICE on asm! for a foreign architecture.

#![crate_type = "rlib"]

use std::arch::asm;

pub unsafe fn aarch64(a: f64, b: f64) -> f64 {
    let c;
    asm!("add {:d}, {:d}, d0", out(vreg) c, in(vreg) a, in("d0") {
        || {};
        b
    });
    //~^^^^ invalid register class
    //~^^^^^ invalid register class
    //~^^^^^^ invalid register
    c
}

pub unsafe fn x86(a: f64, b: f64) -> f64 {
    let c;
    asm!("addsd {}, {}, xmm0", out(xmm_reg) c, in(xmm_reg) a, in("xmm0") b);
    c
}
