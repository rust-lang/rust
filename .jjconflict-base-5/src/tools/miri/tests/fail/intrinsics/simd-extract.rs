#![feature(portable_simd, core_intrinsics)]
use std::simd::*;

fn main() {
    let v = i32x4::splat(0);
    let _x: i32 = unsafe { std::intrinsics::simd::simd_extract(v, 4) };
    //~^ERROR: index 4 is out-of-bounds
}
