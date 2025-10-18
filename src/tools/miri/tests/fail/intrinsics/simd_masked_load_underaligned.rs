#![feature(core_intrinsics, portable_simd)]

use std::intrinsics::simd::*;
use std::simd::*;

fn main() {
    unsafe {
        let buf = Simd::<i32, 5>::splat(0);
        //~v ERROR: accessing memory with alignment
        simd_masked_load::<_, _, _, { SimdAlign::Vector }>(
            i32x4::splat(-1),
            // This is i32-aligned but not i32x4-aligned.
            buf.as_array().as_ptr().offset(1),
            i32x4::splat(0),
        );
    }
}
