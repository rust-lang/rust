//@ only-x86_64

#![feature(repr_simd, never_type)]

use std::arch::{asm, global_asm};

#[repr(simd)]
struct SimdNonCopy([f32; 4]);

fn test1() {
    unsafe {
        // Inputs must be initialized

        let x: u64;
        asm!("{}", in(reg) x);
        //~^ ERROR isn't initialized
        let mut y: u64;
        asm!("{}", inout(reg) y);
        //~^ ERROR isn't initialized
        let _ = y;

        // Outputs require mutable places

        let v: Vec<u64> = vec![0, 1, 2];
        //~^ ERROR is not declared as mutable
        asm!("{}", in(reg) v[0]);
        asm!("{}", out(reg) v[0]);
        asm!("{}", inout(reg) v[0]);
    }
}

fn test2() {
    unsafe {
        // Register operands must be Copy

        asm!("{}", in(xmm_reg) SimdNonCopy([0.0, 0.0, 0.0, 0.0]));
        //~^ ERROR arguments for inline assembly must be copyable

        // Register operands must be integers, floats, SIMD vectors, pointers or
        // function pointers.

        asm!("{}", in(reg) 0i64);
        asm!("{}", in(reg) 0f64);
        asm!("{}", in(xmm_reg) std::arch::x86_64::_mm_setzero_ps());
        asm!("{}", in(reg) 0 as *const u8);
        asm!("{}", in(reg) 0 as *mut u8);
        asm!("{}", in(reg) main as fn());
        asm!("{}", in(reg) |x: i32| x);
        //~^ ERROR cannot use value of type
        asm!("{}", in(reg) vec![0]);
        //~^ ERROR cannot use value of type `Vec<i32>` for inline assembly
        asm!("{}", in(reg) (1, 2, 3));
        //~^ ERROR cannot use value of type `(i32, i32, i32)` for inline assembly
        asm!("{}", in(reg) [1, 2, 3]);
        //~^ ERROR cannot use value of type `[i32; 3]` for inline assembly

        // Register inputs (but not outputs) allow references and function types

        let mut f = main;
        let mut r = &mut 0;
        asm!("{}", in(reg) f);
        asm!("{}", inout(reg) f);
        //~^ ERROR cannot use value of type `fn() {main}` for inline assembly
        asm!("{}", in(reg) r);
        asm!("{}", inout(reg) r);
        //~^ ERROR cannot use value of type `&mut i32` for inline assembly
        let _ = (f, r);

        // Type checks ignore never type

        let u: ! = unreachable!();
        asm!("{}", in(reg) u);
    }
}

fn main() {}
