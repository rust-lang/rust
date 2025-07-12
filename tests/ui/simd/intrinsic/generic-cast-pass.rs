//@ run-pass

#![feature(repr_simd, core_intrinsics)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_cast;

use std::cmp::{max, min};

type V<T> = Simd<T, 2>;

fn main() {
    unsafe {
        let u: V::<u32> = Simd([i16::MIN as u32, i16::MAX as u32]);
        let i: V<i16> = simd_cast(u);
        assert_eq!(i[0], u[0] as i16);
        assert_eq!(i[1], u[1] as i16);
    }

    unsafe {
        let f: V::<f32> = Simd([i16::MIN as f32, i16::MAX as f32]);
        let i: V<i16> = simd_cast(f);
        assert_eq!(i[0], f[0] as i16);
        assert_eq!(i[1], f[1] as i16);
    }

    unsafe {
        let f: V::<f32> = Simd([u8::MIN as f32, u8::MAX as f32]);
        let u: V<u8> = simd_cast(f);
        assert_eq!(u[0], f[0] as u8);
        assert_eq!(u[1], f[1] as u8);
    }

    unsafe {
        // We would like to do isize::MIN..=isize::MAX, but those values are not representable in
        // an f64, so we clamp to the range of an i32 to prevent running into UB.
        let f: V::<f64> = Simd([
            max(isize::MIN, i32::MIN as isize) as f64,
            min(isize::MAX, i32::MAX as isize) as f64,
        ]);
        let i: V<isize> = simd_cast(f);
        assert_eq!(i[0], f[0] as isize);
        assert_eq!(i[1], f[1] as isize);
    }

    unsafe {
        let f: V::<f64> = Simd([
            max(usize::MIN, u32::MIN as usize) as f64,
            min(usize::MAX, u32::MAX as usize) as f64,
        ]);
        let u: V<usize> = simd_cast(f);
        assert_eq!(u[0], f[0] as usize);
        assert_eq!(u[1], f[1] as usize);
    }
}
