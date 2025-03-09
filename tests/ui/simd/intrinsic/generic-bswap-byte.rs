//@ run-pass
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

use std::intrinsics::simd::simd_bswap;

#[repr(simd)]
#[derive(Copy, Clone)]
struct i8x4([i8; 4]);
impl i8x4 {
    fn to_array(self) -> [i8; 4] { unsafe { std::mem::transmute(self) } }
}

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x4([u8; 4]);
impl u8x4 {
    fn to_array(self) -> [u8; 4] { unsafe { std::mem::transmute(self) } }
}

fn main() {
    unsafe {
        assert_eq!(simd_bswap(i8x4([0, 1, 2, 3])).to_array(), [0, 1, 2, 3]);
        assert_eq!(simd_bswap(u8x4([0, 1, 2, 3])).to_array(), [0, 1, 2, 3]);
    }
}
