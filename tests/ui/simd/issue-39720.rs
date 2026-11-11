//@ run-pass
//@ ignore-backends: gcc

#![feature(repr_simd, core_intrinsics)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

pub type Char3 = Simd<i8, 3>;

pub type Short3 = Simd<i16, 3>;

fn main() {
    let cast: Short3 = unsafe { std::intrinsics::simd::simd_cast(Char3::from_array([10, -3, -9])) };

    println!("{:?}", cast);
}
