//@ needs-asm-support
#![feature(repr_simd)]

use std::arch::asm;

#[repr(simd)]
//~^ ERROR attribute should be applied to a struct
//~| ERROR unsupported representation for zero-variant enum
enum Es {}

fn main() {
    unsafe {
        let mut x: Es;
        asm!("{}", out(reg) x);
    }
}
