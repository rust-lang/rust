//@ only-x86_64

#![feature(never_type)]

use std::arch::asm;

fn main() {
    unsafe {
        // Inputs must be initialized

        let x: u64;
        asm!("{}", in(reg) x);
        //~^ ERROR E0381
        let mut y: u64;
        asm!("{}", inout(reg) y);
        //~^ ERROR E0381
        let _ = y;

        // Outputs require mutable places

        let v: Vec<u64> = vec![0, 1, 2];
        //~^ ERROR cannot borrow `v` as mutable, as it is not declared as mutable
        asm!("{}", in(reg) v[0]);
        asm!("{}", out(reg) v[0]);
        asm!("{}", inout(reg) v[0]);

        // Sym operands must point to a function or static

        const C: i32 = 0;
        static S: i32 = 0;
        asm!("{}", sym S);
        asm!("{}", sym main);

        // Register operands must be Copy

        // Register operands must be integers, floats, SIMD vectors, pointers or
        // function pointers.

        asm!("{}", in(reg) 0i64);
        asm!("{}", in(reg) 0f64);
        asm!("{}", in(xmm_reg) std::arch::x86_64::_mm_setzero_ps());
        asm!("{}", in(reg) 0 as *const u8);
        asm!("{}", in(reg) 0 as *mut u8);
        asm!("{}", in(reg) main as fn());

        // Register inputs (but not outputs) allow references and function types

        let mut f = main;
        let mut r = &mut 0;
        asm!("{}", in(reg) f);
        asm!("{}", in(reg) r);
        let _ = (f, r);

        // Type checks ignore never type

        let u: ! = unreachable!();
        asm!("{}", in(reg) u);
    }
}
