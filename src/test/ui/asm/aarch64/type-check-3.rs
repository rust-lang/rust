// only-aarch64
// compile-flags: -C target-feature=+neon

#![feature(asm, global_asm, repr_simd, stdsimd)]

use std::arch::aarch64::float64x2_t;

#[repr(simd)]
#[derive(Copy, Clone)]
struct Simd256bit(f64, f64,f64, f64);

fn main() {
    let f64x2: float64x2_t = unsafe { std::mem::transmute(0i128) };
    let f64x4 = Simd256bit(0.0, 0.0, 0.0, 0.0);

    unsafe {
        // Types must be listed in the register class.

        // Success cases
        asm!("{:w}", in(reg) 0u8);
        asm!("{:w}", in(reg) 0u16);
        asm!("{:w}", in(reg) 0u32);
        asm!("{:w}", in(reg) 0f32);
        asm!("{}", in(reg) 0i64);
        asm!("{}", in(reg) 0f64);

        asm!("{:b}", in(vreg) 0u8);
        asm!("{:h}", in(vreg) 0u16);
        asm!("{:s}", in(vreg) 0u32);
        asm!("{:s}", in(vreg) 0f32);
        asm!("{:d}", in(vreg) 0u64);
        asm!("{:d}", in(vreg) 0f64);
        asm!("{:q}", in(vreg) f64x2);
        asm!("{:v}", in(vreg) f64x2);

        // Should be the same as vreg
        asm!("{:q}", in(vreg_low16) f64x2);

        // Template modifiers of a different size to the argument are fine
        asm!("{:w}", in(reg) 0u64);
        asm!("{:x}", in(reg) 0u32);
        asm!("{:b}", in(vreg) 0u64);
        asm!("{:d}", in(vreg_low16) f64x2);


        // Template modifier suggestions for sub-registers

        asm!("{}", in(reg) 0u8);
        //~^ WARN formatting may not be suitable for sub-register argument
        asm!("{}", in(reg) 0u16);
        //~^ WARN formatting may not be suitable for sub-register argument
        asm!("{}", in(reg) 0i32);
        //~^ WARN formatting may not be suitable for sub-register argument
        asm!("{}", in(reg) 0f32);
        //~^ WARN formatting may not be suitable for sub-register argument

        asm!("{}", in(vreg) 0i16);
        //~^ WARN formatting may not be suitable for sub-register argument
        asm!("{}", in(vreg) 0f32);
        //~^ WARN formatting may not be suitable for sub-register argument
        asm!("{}", in(vreg) 0f64);
        //~^ WARN formatting may not be suitable for sub-register argument
        asm!("{}", in(vreg_low16) 0f64);
        //~^ WARN formatting may not be suitable for sub-register argument

        asm!("{0} {0}", in(reg) 0i16);
        //~^ WARN formatting may not be suitable for sub-register argument
        asm!("{0} {0:x}", in(reg) 0i16);
        //~^ WARN formatting may not be suitable for sub-register argument

        // Invalid registers

        asm!("{}", in(reg) 0i128);
        //~^ ERROR type `i128` cannot be used with this register class
        asm!("{}", in(reg) f64x2);
        //~^ ERROR type `float64x2_t` cannot be used with this register class
        asm!("{}", in(vreg) f64x4);
        //~^ ERROR type `Simd256bit` cannot be used with this register class

        // Split inout operands must have compatible types

        let mut val_i16: i16;
        let mut val_f32: f32;
        let mut val_u32: u32;
        let mut val_u64: u64;
        let mut val_ptr: *mut u8;
        asm!("{:x}", inout(reg) 0u16 => val_i16);
        asm!("{:x}", inout(reg) 0u32 => val_f32);
        //~^ ERROR incompatible types for asm inout argument
        asm!("{:x}", inout(reg) 0u32 => val_ptr);
        //~^ ERROR incompatible types for asm inout argument
        asm!("{:x}", inout(reg) main => val_u32);
        //~^ ERROR incompatible types for asm inout argument
        asm!("{:x}", inout(reg) 0u64 => val_ptr);
        asm!("{:x}", inout(reg) main => val_u64);
    }
}

// Constants must be... constant

static S: i32 = 1;
const fn const_foo(x: i32) -> i32 {
    x
}
const fn const_bar<T>(x: T) -> T {
    x
}
global_asm!("{}", const S);
//~^ ERROR constants cannot refer to statics
global_asm!("{}", const const_foo(0));
global_asm!("{}", const const_foo(S));
//~^ ERROR constants cannot refer to statics
global_asm!("{}", const const_bar(0));
global_asm!("{}", const const_bar(S));
//~^ ERROR constants cannot refer to statics
