//@ only-aarch64

#![feature(repr_simd, never_type)]

use std::arch::{asm, global_asm};

#[repr(simd)]
#[derive(Clone, Copy)]
struct SimdType([f32; 4]);

#[repr(simd)]
struct SimdNonCopy([f32; 4]);

fn main() {
    unsafe {
        // Inputs must be initialized

        // Register operands must be Copy

        asm!("{:v}", in(vreg) SimdNonCopy([0.0, 0.0, 0.0, 0.0]));
        //~^ ERROR arguments for inline assembly must be copyable

        // Register operands must be integers, floats, SIMD vectors, pointers or
        // function pointers.

        asm!("{}", in(reg) 0i64);
        asm!("{}", in(reg) 0f64);
        asm!("{:v}", in(vreg) SimdType([0.0, 0.0, 0.0, 0.0]));
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
