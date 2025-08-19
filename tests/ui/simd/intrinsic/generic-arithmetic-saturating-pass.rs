//@ run-pass
//@ ignore-emscripten

#![allow(non_camel_case_types)]
#![feature(repr_simd, core_intrinsics)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{simd_saturating_add, simd_saturating_sub};

type I32<const N: usize> = Simd<i32, N>;

fn main() {
    // unsigned
    {
        const M: u32 = u32::MAX;

        let a = u32x4::from_array([1, 2, 3, 4]);
        let b = u32x4::from_array([2, 4, 6, 8]);
        let m = u32x4::from_array([M, M, M, M]);
        let m1 = u32x4::from_array([M - 1, M - 1, M - 1, M - 1]);
        let z = u32x4::from_array([0, 0, 0, 0]);

        unsafe {
            assert_eq!(simd_saturating_add(z, z), z);
            assert_eq!(simd_saturating_add(z, a), a);
            assert_eq!(simd_saturating_add(b, z), b);
            assert_eq!(simd_saturating_add(a, a), b);
            assert_eq!(simd_saturating_add(a, m), m);
            assert_eq!(simd_saturating_add(m, b), m);
            assert_eq!(simd_saturating_add(m1, a), m);

            assert_eq!(simd_saturating_sub(b, z), b);
            assert_eq!(simd_saturating_sub(b, a), a);
            assert_eq!(simd_saturating_sub(a, a), z);
            assert_eq!(simd_saturating_sub(a, b), z);
            assert_eq!(simd_saturating_sub(a, m1), z);
            assert_eq!(simd_saturating_sub(b, m1), z);
        }
    }

    // signed
    {
        const MIN: i32 = i32::MIN;
        const MAX: i32 = i32::MAX;

        let a = I32::<4>::from_array([1, 2, 3, 4]);
        let b = I32::<4>::from_array([2, 4, 6, 8]);
        let c = I32::<4>::from_array([-1, -2, -3, -4]);
        let d = I32::<4>::from_array([-2, -4, -6, -8]);

        let max = I32::<4>::from_array([MAX, MAX, MAX, MAX]);
        let max1 = I32::<4>::from_array([MAX - 1, MAX - 1, MAX - 1, MAX - 1]);
        let min = I32::<4>::from_array([MIN, MIN, MIN, MIN]);
        let min1 = I32::<4>::from_array([MIN + 1, MIN + 1, MIN + 1, MIN + 1]);

        let z = I32::<4>::from_array([0, 0, 0, 0]);

        unsafe {
            assert_eq!(simd_saturating_add(z, z), z);
            assert_eq!(simd_saturating_add(z, a), a);
            assert_eq!(simd_saturating_add(b, z), b);
            assert_eq!(simd_saturating_add(a, a), b);
            assert_eq!(simd_saturating_add(a, max), max);
            assert_eq!(simd_saturating_add(max, b), max);
            assert_eq!(simd_saturating_add(max1, a), max);
            assert_eq!(simd_saturating_add(min1, z), min1);
            assert_eq!(simd_saturating_add(min, z), min);
            assert_eq!(simd_saturating_add(min1, c), min);
            assert_eq!(simd_saturating_add(min, c), min);
            assert_eq!(simd_saturating_add(min1, d), min);
            assert_eq!(simd_saturating_add(min, d), min);

            assert_eq!(simd_saturating_sub(b, z), b);
            assert_eq!(simd_saturating_sub(b, a), a);
            assert_eq!(simd_saturating_sub(a, a), z);
            assert_eq!(simd_saturating_sub(a, b), c);
            assert_eq!(simd_saturating_sub(z, max), min1);
            assert_eq!(simd_saturating_sub(min1, z), min1);
            assert_eq!(simd_saturating_sub(min1, a), min);
            assert_eq!(simd_saturating_sub(min1, b), min);
        }
    }
}
