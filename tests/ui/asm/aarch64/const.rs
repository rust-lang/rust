//@ only-aarch64
//@ run-pass
//@ needs-asm-support

use std::arch::{asm, global_asm};

fn const_generic<const X: usize>() -> usize {
    unsafe {
        let a: usize;
        asm!("mov {}, {}", out(reg) a, const X);
        a
    }
}

const fn constfn(x: usize) -> usize {
    x
}

fn main() {
    unsafe {
        let a: usize;
        asm!("mov {}, {}", out(reg) a, const 5);
        assert_eq!(a, 5);

        let b: usize;
        asm!("mov {}, {}", out(reg) b, const constfn(5));
        assert_eq!(b, 5);

        let c: usize;
        asm!("mov {}, {}", out(reg) c, const constfn(5) + constfn(5));
        assert_eq!(c, 10);
    }

    let d = const_generic::<5>();
    assert_eq!(d, 5);
}

global_asm!("mov x0, {}", const 5);
global_asm!("mov x0, {}", const constfn(5));
global_asm!("mov x0, {}", const constfn(5) + constfn(5));
