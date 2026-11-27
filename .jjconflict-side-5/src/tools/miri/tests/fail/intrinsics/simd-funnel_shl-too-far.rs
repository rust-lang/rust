#![feature(core_intrinsics, portable_simd)]

use std::intrinsics::simd::simd_funnel_shl;
use std::simd::*;

fn main() {
    unsafe {
        let x = i32x2::from_array([1, 1]);
        let y = i32x2::from_array([100, 0]);
        simd_funnel_shl(x, x, y); //~ERROR: overflowing shift by 100 in `simd_funnel_shl` in lane 0
    }
}
