//@ run-pass
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_bswap;

fn main() {
    unsafe {
        assert_eq!(simd_bswap(i8x4::from_array([0, 1, 2, 3])).into_array(), [0, 1, 2, 3]);
        assert_eq!(simd_bswap(u8x4::from_array([0, 1, 2, 3])).into_array(), [0, 1, 2, 3]);
    }
}
