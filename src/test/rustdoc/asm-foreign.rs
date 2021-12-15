// Make sure rustdoc accepts asm! for a foreign architecture.

use std::arch::asm;

// @has asm_foreign/fn.aarch64.html
pub unsafe fn aarch64(a: f64, b: f64) -> f64 {
    let c;
    asm!("add {:d}, {:d}, d0", out(vreg) c, in(vreg) a, in("d0") {
        || {};
        b
    });
    c
}

// @has asm_foreign/fn.x86.html
pub unsafe fn x86(a: f64, b: f64) -> f64 {
    let c;
    asm!("addsd {}, {}, xmm0", out(xmm_reg) c, in(xmm_reg) a, in("xmm0") b);
    c
}
