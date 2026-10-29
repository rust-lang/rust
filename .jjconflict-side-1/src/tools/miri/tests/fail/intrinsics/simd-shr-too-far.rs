#![feature(core_intrinsics, repr_simd)]

use std::intrinsics::simd::simd_shr;

#[repr(simd)]
#[allow(non_camel_case_types)]
struct i32x2([i32; 2]);

fn main() {
    unsafe {
        let x = i32x2([1, 1]);
        let y = i32x2([20, 40]);
        simd_shr(x, y); //~ERROR: overflowing shift by 40 in `simd_shr` in lane 1
    }
}
