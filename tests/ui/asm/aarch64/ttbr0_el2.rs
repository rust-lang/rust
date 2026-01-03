//! Regression test for #97724, recognising ttbr0_el2 as a valid armv8 system register
//@ only-aarch64
//@ build-pass
use std::arch::asm;

static PT: [u64; 512] = [0; 512];
fn main() {
    unsafe {
        asm!("msr ttbr0_el2, {pt}", pt = in(reg) &PT as *const _ );
    }
}
