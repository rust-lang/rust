#![feature(portable_simd)]
use std::simd::*;

fn main() {
    let a = f32x4::splat(10.0);
    let b = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(a + b, f32x4::from_array([11.0, 12.0, 13.0, 14.0]));
}
