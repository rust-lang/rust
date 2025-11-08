#![feature(core_intrinsics, portable_simd)]

use std::intrinsics::simd::simd_funnel_shr;
use std::simd::*;

fn main() {
    unsafe {
        let x = i32x2::from_array([1, 1]);
        let y = i32x2::from_array([20, 40]);
        simd_funnel_shr(x, x, y); //~ERROR: overflowing shift by 40 in `simd_funnel_shr` in lane 1
    }
}
