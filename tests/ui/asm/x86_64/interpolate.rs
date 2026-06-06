//@ only-x86_64
//@ run-pass
//@ needs-asm-support

#![feature(asm_interpolate)]

use std::arch::asm;

trait Interpolate {
    const OP: &str;
}

impl Interpolate for usize {
    const OP: &str = "mov";
}

fn main() {
    unsafe {
        let a: usize;
        asm!("{} {}, {}", interpolate "mov", out(reg) a, const 5);
        assert_eq!(a, 5);

        const MOV: &str = "mov";
        let b: usize;
        asm!("{} {}, {}", interpolate MOV, out(reg) b, const 6);
        assert_eq!(b, 6);

        let c: usize;
        asm!("{} {}, {}", interpolate usize::OP, out(reg) c, const 7);
        assert_eq!(c, 7);
    }
}
