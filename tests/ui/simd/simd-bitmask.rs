//@run-pass
#![feature(repr_simd, core_intrinsics)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{simd_bitmask, simd_select_bitmask};

fn main() {
    unsafe {
        let v = Simd::<i8, 4>([-1, 0, -1, 0]);
        let i: u8 = simd_bitmask(v);
        let a: [u8; 1] = simd_bitmask(v);

        if cfg!(target_endian = "little") {
            assert_eq!(i, 0b0101);
            assert_eq!(a, [0b0101]);
        } else {
            assert_eq!(i, 0b1010);
            assert_eq!(a, [0b1010]);
        }

        let v = Simd::<i8, 16>([0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0]);
        let i: u16 = simd_bitmask(v);
        let a: [u8; 2] = simd_bitmask(v);

        if cfg!(target_endian = "little") {
            assert_eq!(i, 0b0101000000001100);
            assert_eq!(a, [0b00001100, 0b01010000]);
        } else {
            assert_eq!(i, 0b0011000000001010);
            assert_eq!(a, [0b00110000, 0b00001010]);
        }
    }

    unsafe {
        let a = Simd::<i32, 4>([0, 1, 2, 3]);
        let b = Simd::<i32, 4>([8, 9, 10, 11]);
        let e = [0, 9, 2, 11];

        let mask = if cfg!(target_endian = "little") { 0b0101u8 } else { 0b1010u8 };
        let r = simd_select_bitmask(mask, a, b);
        assert_eq!(r.into_array(), e);

        let mask = if cfg!(target_endian = "little") { [0b0101u8] } else { [0b1010u8] };
        let r = simd_select_bitmask(mask, a, b);
        assert_eq!(r.into_array(), e);

        let a = Simd::<i32, 16>([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let b = Simd::<i32, 16>([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]);
        let e = [16, 17, 2, 3, 20, 21, 22, 23, 24, 25, 26, 27, 12, 29, 14, 31];

        let mask = if cfg!(target_endian = "little") {
            0b0101000000001100u16
        } else {
            0b0011000000001010u16
        };
        let r = simd_select_bitmask(mask, a, b);
        assert_eq!(r.into_array(), e);

        let mask = if cfg!(target_endian = "little") {
            [0b00001100u8, 0b01010000u8]
        } else {
            [0b00110000u8, 0b00001010u8]
        };
        let r = simd_select_bitmask(mask, a, b);
        assert_eq!(r.into_array(), e);
    }
}
