//@run-pass
// FIXME: broken codegen on big-endian (https://github.com/rust-lang/rust/issues/127205)
// This should be merged into `simd-bitmask` once that's fixed.
//@ ignore-endian-big
#![feature(repr_simd, core_intrinsics)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{simd_bitmask, simd_select_bitmask};

fn main() {
    // Non-power-of-2 multi-byte mask.
    #[allow(non_camel_case_types)]
    type i32x10 = PackedSimd<i32, 10>;
    impl i32x10 {
        fn splat(x: i32) -> Self {
            Self([x; 10])
        }
    }
    unsafe {
        let mask = i32x10::from_array([!0, !0, 0, !0, 0, 0, !0, 0, !0, 0]);
        let mask_bits = if cfg!(target_endian = "little") { 0b0101001011 } else { 0b1101001010 };
        let mask_bytes =
            if cfg!(target_endian = "little") { [0b01001011, 0b01] } else { [0b11, 0b01001010] };

        let bitmask1: u16 = simd_bitmask(mask);
        let bitmask2: [u8; 2] = simd_bitmask(mask);
        assert_eq!(bitmask1, mask_bits);
        assert_eq!(bitmask2, mask_bytes);

        let selected1 = simd_select_bitmask::<u16, _>(
            mask_bits,
            i32x10::splat(!0), // yes
            i32x10::splat(0),  // no
        );
        let selected2 = simd_select_bitmask::<[u8; 2], _>(
            mask_bytes,
            i32x10::splat(!0), // yes
            i32x10::splat(0),  // no
        );
        assert_eq!(selected1, mask);
        assert_eq!(selected2, mask);
    }

    // Test for a mask where the next multiple of 8 is not a power of two.
    #[allow(non_camel_case_types)]
    type i32x20 = PackedSimd<i32, 20>;
    impl i32x20 {
        fn splat(x: i32) -> Self {
            Self([x; 20])
        }
    }
    unsafe {
        let mask = i32x20::from_array([
            !0, !0,  0, !0,  0,
             0, !0,  0, !0,  0,
             0,  0,  0, !0, !0,
            !0, !0, !0, !0, !0,
        ]);
        let mask_bits = if cfg!(target_endian = "little") {
            0b11111110000101001011
        } else {
            0b11010010100001111111
        };
        let mask_bytes = if cfg!(target_endian = "little") {
            [0b01001011, 0b11100001, 0b1111]
        } else {
            [0b1101, 0b00101000, 0b01111111]
        };

        let bitmask1: u32 = simd_bitmask(mask);
        let bitmask2: [u8; 3] = simd_bitmask(mask);
        assert_eq!(bitmask1, mask_bits);
        assert_eq!(bitmask2, mask_bytes);

        let selected1 = simd_select_bitmask::<u32, _>(
            mask_bits,
            i32x20::splat(!0), // yes
            i32x20::splat(0),  // no
        );
        let selected2 = simd_select_bitmask::<[u8; 3], _>(
            mask_bytes,
            i32x20::splat(!0), // yes
            i32x20::splat(0),  // no
        );
        assert_eq!(selected1, mask);
        assert_eq!(selected2, mask);
    }
}
