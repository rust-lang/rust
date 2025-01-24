#![feature(core_intrinsics, repr_simd)]

use std::intrinsics::simd::simd_select_bitmask;

#[repr(simd)]
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
struct i32x2([i32; 2]);

fn main() {
    unsafe {
        let x = i32x2([0, 1]);
        simd_select_bitmask(0b11111111u8, x, x); //~ERROR: bitmask less than 8 bits long must be filled with 0s for the remaining bits
    }
}
