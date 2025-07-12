//@ run-pass
#![feature(repr_simd, core_intrinsics)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_cast;

type V<T> = Simd<T, 4>;

fn main() {
    let u: V::<usize> = Simd([0, 1, 2, 3]);
    let uu32: V<u32> = unsafe { simd_cast(u) };
    let ui64: V<i64> = unsafe { simd_cast(u) };

    for (u, (uu32, ui64)) in u
        .as_array()
        .iter()
        .zip(uu32.as_array().iter().zip(ui64.as_array().iter()))
    {
        assert_eq!(*u as u32, *uu32);
        assert_eq!(*u as i64, *ui64);
    }
}
