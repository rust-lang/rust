//@ run-pass
//@ ignore-emscripten
//@ ignore-endian-big behavior of simd_bitmask is endian-specific
//@ compile-flags: --cfg minisimd_const

// Test that the simd_bitmask intrinsic produces correct results.
#![feature(repr_simd, core_intrinsics, const_trait_impl, const_cmp, const_index)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_bitmask;

const fn bitmask() {
    let z = u32x4::from_array([0, 0, 0, 0]);
    let ez = 0_u8;

    let o = u32x4::from_array([!0, !0, !0, !0]);
    let eo = 0b_1111_u8;

    let m0 = u32x4::from_array([!0, 0, !0, 0]);
    let e0 = 0b_0000_0101_u8;

    let e = 0b_1101;

    // Check usize / isize
    let msize = usizex4::from_array([usize::MAX, 0, usize::MAX, usize::MAX]);

    unsafe {
        let r: u8 = simd_bitmask(z);
        assert_eq!(r, ez);

        let r: u8 = simd_bitmask(o);
        assert_eq!(r, eo);

        let r: u8 = simd_bitmask(m0);
        assert_eq!(r, e0);

        let r: u8 = simd_bitmask(msize);
        assert_eq!(r, e);
    }
}

fn main() {
    const { bitmask() };
    bitmask();
}
