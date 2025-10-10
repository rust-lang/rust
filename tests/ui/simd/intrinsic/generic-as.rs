//@ run-pass
//@ ignore-backends: gcc

#![feature(repr_simd, core_intrinsics)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_as;

type V<T> = Simd<T, 2>;

fn main() {
    unsafe {
        let u: V::<u32> = Simd([u32::MIN, u32::MAX]);
        let i: V<i16> = simd_as(u);
        assert_eq!(i[0], u[0] as i16);
        assert_eq!(i[1], u[1] as i16);
    }

    unsafe {
        let f: V::<f32> = Simd([f32::MIN, f32::MAX]);
        let i: V<i16> = simd_as(f);
        assert_eq!(i[0], f[0] as i16);
        assert_eq!(i[1], f[1] as i16);
    }

    unsafe {
        let f: V::<f32> = Simd([f32::MIN, f32::MAX]);
        let u: V<u8> = simd_as(f);
        assert_eq!(u[0], f[0] as u8);
        assert_eq!(u[1], f[1] as u8);
    }

    unsafe {
        let f: V::<f64> = Simd([f64::MIN, f64::MAX]);
        let i: V<isize> = simd_as(f);
        assert_eq!(i[0], f[0] as isize);
        assert_eq!(i[1], f[1] as isize);
    }

    unsafe {
        let f: V::<f64> = Simd([f64::MIN, f64::MAX]);
        let u: V<usize> = simd_as(f);
        assert_eq!(u[0], f[0] as usize);
        assert_eq!(u[1], f[1] as usize);
    }
}
