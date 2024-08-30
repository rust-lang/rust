#![feature(core_intrinsics, repr_simd)]

use std::intrinsics::simd::simd_reduce_any;

#[repr(simd)]
#[allow(non_camel_case_types)]
struct i32x2([i32; 2]);

fn main() {
    unsafe {
        let x = i32x2([0, 1]);
        simd_reduce_any(x); //~ERROR: must be all-0-bits or all-1-bits
    }
}
