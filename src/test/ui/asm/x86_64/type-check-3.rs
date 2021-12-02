// only-x86_64
// compile-flags: -C target-feature=+avx512f

#![feature(asm, global_asm, asm_const)]

use std::arch::x86_64::{_mm256_setzero_ps, _mm_setzero_ps};

fn main() {
    unsafe {
        // Types must be listed in the register class.

        asm!("{}", in(reg) 0i128);
        //~^ ERROR type `i128` cannot be used with this register class
        asm!("{}", in(reg) _mm_setzero_ps());
        //~^ ERROR type `__m128` cannot be used with this register class
        asm!("{}", in(reg) _mm256_setzero_ps());
        //~^ ERROR type `__m256` cannot be used with this register class
        asm!("{}", in(xmm_reg) 0u8);
        //~^ ERROR type `u8` cannot be used with this register class
        asm!("{:e}", in(reg) 0i32);
        asm!("{}", in(xmm_reg) 0i32);
        asm!("{:e}", in(reg) 0f32);
        asm!("{}", in(xmm_reg) 0f32);
        asm!("{}", in(xmm_reg) _mm_setzero_ps());
        asm!("{:x}", in(ymm_reg) _mm_setzero_ps());
        asm!("{}", in(kreg) 0u16);
        asm!("{}", in(kreg) 0u64);
        //~^ ERROR `avx512bw` target feature is not enabled

        // Template modifier suggestions for sub-registers

        asm!("{0} {0}", in(reg) 0i16);
        //~^ WARN formatting may not be suitable for sub-register argument
        asm!("{0} {0:x}", in(reg) 0i16);
        //~^ WARN formatting may not be suitable for sub-register argument
        asm!("{}", in(reg) 0i32);
        //~^ WARN formatting may not be suitable for sub-register argument
        asm!("{}", in(reg) 0i64);
        asm!("{}", in(ymm_reg) 0i64);
        //~^ WARN formatting may not be suitable for sub-register argument
        asm!("{}", in(ymm_reg) _mm256_setzero_ps());
        asm!("{:l}", in(reg) 0i16);
        asm!("{:l}", in(reg) 0i32);
        asm!("{:l}", in(reg) 0i64);
        asm!("{:x}", in(ymm_reg) 0i64);
        asm!("{:x}", in(ymm_reg) _mm256_setzero_ps());

        // Suggest different register class for type

        asm!("{}", in(reg) 0i8);
        //~^ ERROR type `i8` cannot be used with this register class
        asm!("{}", in(reg_byte) 0i8);

        // Split inout operands must have compatible types

        let mut val_i16: i16;
        let mut val_f32: f32;
        let mut val_u32: u32;
        let mut val_u64: u64;
        let mut val_ptr: *mut u8;
        asm!("{:r}", inout(reg) 0u16 => val_i16);
        asm!("{:r}", inout(reg) 0u32 => val_f32);
        //~^ ERROR incompatible types for asm inout argument
        asm!("{:r}", inout(reg) 0u32 => val_ptr);
        //~^ ERROR incompatible types for asm inout argument
        asm!("{:r}", inout(reg) main => val_u32);
        //~^ ERROR incompatible types for asm inout argument
        asm!("{:r}", inout(reg) 0u64 => val_ptr);
        asm!("{:r}", inout(reg) main => val_u64);
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
