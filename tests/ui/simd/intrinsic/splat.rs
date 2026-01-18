//@ run-pass
#![feature(repr_simd, core_intrinsics)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_splat;

fn main() {
    unsafe {
        let x: Simd<u32, 1> = simd_splat(123u32);
        let y: Simd<u32, 1> = const { simd_splat(123u32) };
        assert_eq!(x.into_array(), [123; 1]);
        assert_eq!(x.into_array(), y.into_array());

        let x: u16x2 = simd_splat(42u16);
        let y: u16x2 = const { simd_splat(42u16) };
        assert_eq!(x.into_array(), [42; 2]);
        assert_eq!(x.into_array(), y.into_array());

        let x: u128x4 = simd_splat(42u128);
        let y: u128x4 = const { simd_splat(42u128) };
        assert_eq!(x.into_array(), [42; 4]);
        assert_eq!(x.into_array(), y.into_array());

        let x: i32x4 = simd_splat(-7i32);
        let y: i32x4 = const { simd_splat(-7i32) };
        assert_eq!(x.into_array(), [-7; 4]);
        assert_eq!(x.into_array(), y.into_array());

        let x: f32x4 = simd_splat(42.0f32);
        let y: f32x4 = const { simd_splat(42.0f32) };
        assert_eq!(x.into_array(), [42.0; 4]);
        assert_eq!(x.into_array(), y.into_array());

        let x: f64x2 = simd_splat(42.0f64);
        let y: f64x2 = const { simd_splat(42.0f64) };
        assert_eq!(x.into_array(), [42.0; 2]);
        assert_eq!(x.into_array(), y.into_array());

        static ZERO: u8 = 0u8;
        let x: Simd<*const u8, 2> = simd_splat(&raw const ZERO);
        assert_eq!(x.into_array(), [&raw const ZERO; 2]);

        // FIXME: this hits "could not evaluate shuffle_indices at compile time",
        // emitted in `immediate_const_vector`. const-eval should be able to handle
        // this though I think? `const { [&raw const ZERO; 2] }` appears to work.
        // let y: Simd<*const u8, 2> = const { simd_splat(&raw const ZERO) };
        // assert_eq!(x.into_array(), y.into_array());
    }
}
