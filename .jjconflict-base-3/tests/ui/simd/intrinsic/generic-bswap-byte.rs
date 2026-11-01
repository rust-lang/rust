//@ run-pass
//@ compile-flags: --cfg minisimd_const
#![feature(repr_simd, core_intrinsics, const_trait_impl, const_cmp, const_index)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_bswap;

const fn bswap() {
    unsafe {
        assert_eq!(simd_bswap(i8x4::from_array([0, 1, 2, 3])).into_array(), [0, 1, 2, 3]);
        assert_eq!(simd_bswap(u8x4::from_array([0, 1, 2, 3])).into_array(), [0, 1, 2, 3]);
    }
}

fn main() {
    const { bswap() };
    bswap();
}
