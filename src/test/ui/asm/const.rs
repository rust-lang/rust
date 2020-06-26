// no-system-llvm
// only-x86_64
// run-pass

#![feature(asm)]

use std::mem::size_of;

trait Proj {
    const C: usize;
}
impl Proj for i8 {
    const C: usize = 8;
}
impl Proj for i16 {
    const C: usize = 16;
}

const fn constfn(x: usize) -> usize {
    x
}

fn generic<T: Proj>() {
    unsafe {
        let a: usize;
        asm!("mov {}, {}", out(reg) a, const size_of::<T>());
        assert_eq!(a, size_of::<T>());

        let b: usize;
        asm!("mov {}, {}", out(reg) b, const size_of::<T>() + constfn(5));
        assert_eq!(b, size_of::<T>() + 5);

        let c: usize;
        asm!("mov {}, {}", out(reg) c, const T::C);
        assert_eq!(c, T::C);
    }
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

    generic::<i8>();
    generic::<i16>();
}
