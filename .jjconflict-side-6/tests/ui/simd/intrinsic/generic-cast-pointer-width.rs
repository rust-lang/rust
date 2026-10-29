//@ run-pass
//@ compile-flags: --cfg minisimd_const
#![feature(repr_simd, core_intrinsics, const_trait_impl, const_cmp, const_index)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_cast;

type V<T> = Simd<T, 4>;

const fn cast_ptr_width() {
    let u: V::<usize> = Simd([0, 1, 2, 3]);
    let uu32: V<u32> = unsafe { simd_cast(u) };
    let ui64: V<i64> = unsafe { simd_cast(u) };

    assert_eq!(uu32, V::<u32>::from_array([0, 1, 2, 3]));
    assert_eq!(ui64, V::<i64>::from_array([0, 1, 2, 3]));
}

fn main() {
    const { cast_ptr_width() };
    cast_ptr_width();
}
