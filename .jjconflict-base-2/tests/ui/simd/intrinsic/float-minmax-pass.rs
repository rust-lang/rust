//@ run-pass
//@ ignore-emscripten
//@ compile-flags: --cfg minisimd_const -O

// Test that the simd_{min,max}imum_number_nsz intrinsics produce the correct results.

#![feature(repr_simd, core_intrinsics, const_trait_impl, const_cmp, const_index)]
#![allow(non_camel_case_types)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::*;
use std::hint::black_box;

const fn minmax() {
    let x = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = f32x4::from_array([2.0, 1.0, 4.0, 3.0]);

    let nan = f32::NAN;
    // The "-1" works because we rely on `NAN` to have an all-0 payload, so the signaling
    // bit is the least significant non-zero bit.
    let snan = f32::from_bits(f32::NAN.to_bits() - 1);

    let n = f32x4::from_array([nan, nan, snan, snan]);

    unsafe {
        let min0 = simd_minimum_number_nsz(x, y);
        let min1 = simd_minimum_number_nsz(y, black_box(x));
        assert_eq!(min0, min1);
        let e = f32x4::from_array([1.0, 1.0, 3.0, 3.0]);
        assert_eq!(min0, e);
        let minn = simd_minimum_number_nsz(x, n);
        assert_eq!(minn, x);
        let minn = simd_minimum_number_nsz(black_box(y), n);
        assert_eq!(minn, y);

        let max0 = simd_maximum_number_nsz(x, y);
        let max1 = simd_maximum_number_nsz(y, black_box(x));
        assert_eq!(max0, max1);
        let e = f32x4::from_array([2.0, 2.0, 4.0, 4.0]);
        assert_eq!(max0, e);
        let maxn = simd_maximum_number_nsz(x, n);
        assert_eq!(maxn, x);
        let maxn = simd_maximum_number_nsz(n, black_box(y));
        assert_eq!(maxn, y);
    }
}

fn main() {
    const { minmax() };
    minmax();
}
