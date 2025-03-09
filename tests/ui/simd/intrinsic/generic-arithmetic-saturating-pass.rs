//@ run-pass
//@ ignore-emscripten

#![allow(non_camel_case_types)]
#![feature(repr_simd, core_intrinsics)]

use std::intrinsics::simd::{simd_saturating_add, simd_saturating_sub};

#[repr(simd)]
#[derive(Copy, Clone)]
struct u32x4(pub [u32; 4]);
impl u32x4 {
    fn to_array(self) -> [u32; 4] { unsafe { std::mem::transmute(self) } }
}

#[repr(simd)]
#[derive(Copy, Clone)]
struct I32<const N: usize>([i32; N]);
impl<const N: usize> I32<N> {
    fn to_array(self) -> [i32; N] { unsafe { std::intrinsics::transmute_unchecked(self) } }
}

macro_rules! all_eq {
    ($a: expr, $b: expr) => {{
        let a = $a;
        let b = $b;
        assert_eq!(a.to_array(), b.to_array());
    }};
}

fn main() {
    // unsigned
    {
        const M: u32 = u32::MAX;

        let a = u32x4([1, 2, 3, 4]);
        let b = u32x4([2, 4, 6, 8]);
        let m = u32x4([M, M, M, M]);
        let m1 = u32x4([M - 1, M - 1, M - 1, M - 1]);
        let z = u32x4([0, 0, 0, 0]);

        unsafe {
            all_eq!(simd_saturating_add(z, z), z);
            all_eq!(simd_saturating_add(z, a), a);
            all_eq!(simd_saturating_add(b, z), b);
            all_eq!(simd_saturating_add(a, a), b);
            all_eq!(simd_saturating_add(a, m), m);
            all_eq!(simd_saturating_add(m, b), m);
            all_eq!(simd_saturating_add(m1, a), m);

            all_eq!(simd_saturating_sub(b, z), b);
            all_eq!(simd_saturating_sub(b, a), a);
            all_eq!(simd_saturating_sub(a, a), z);
            all_eq!(simd_saturating_sub(a, b), z);
            all_eq!(simd_saturating_sub(a, m1), z);
            all_eq!(simd_saturating_sub(b, m1), z);
        }
    }

    // signed
    {
        const MIN: i32 = i32::MIN;
        const MAX: i32 = i32::MAX;

        let a = I32::<4>([1, 2, 3, 4]);
        let b = I32::<4>([2, 4, 6, 8]);
        let c = I32::<4>([-1, -2, -3, -4]);
        let d = I32::<4>([-2, -4, -6, -8]);

        let max = I32::<4>([MAX, MAX, MAX, MAX]);
        let max1 = I32::<4>([MAX - 1, MAX - 1, MAX - 1, MAX - 1]);
        let min = I32::<4>([MIN, MIN, MIN, MIN]);
        let min1 = I32::<4>([MIN + 1, MIN + 1, MIN + 1, MIN + 1]);

        let z = I32::<4>([0, 0, 0, 0]);

        unsafe {
            all_eq!(simd_saturating_add(z, z), z);
            all_eq!(simd_saturating_add(z, a), a);
            all_eq!(simd_saturating_add(b, z), b);
            all_eq!(simd_saturating_add(a, a), b);
            all_eq!(simd_saturating_add(a, max), max);
            all_eq!(simd_saturating_add(max, b), max);
            all_eq!(simd_saturating_add(max1, a), max);
            all_eq!(simd_saturating_add(min1, z), min1);
            all_eq!(simd_saturating_add(min, z), min);
            all_eq!(simd_saturating_add(min1, c), min);
            all_eq!(simd_saturating_add(min, c), min);
            all_eq!(simd_saturating_add(min1, d), min);
            all_eq!(simd_saturating_add(min, d), min);

            all_eq!(simd_saturating_sub(b, z), b);
            all_eq!(simd_saturating_sub(b, a), a);
            all_eq!(simd_saturating_sub(a, a), z);
            all_eq!(simd_saturating_sub(a, b), c);
            all_eq!(simd_saturating_sub(z, max), min1);
            all_eq!(simd_saturating_sub(min1, z), min1);
            all_eq!(simd_saturating_sub(min1, a), min);
            all_eq!(simd_saturating_sub(min1, b), min);
        }
    }
}
