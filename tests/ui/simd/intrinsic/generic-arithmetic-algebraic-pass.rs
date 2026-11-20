//@ run-pass
//@ compile-flags: --cfg minisimd_const

#![allow(non_camel_case_types)]
#![feature(repr_simd, core_intrinsics, const_trait_impl, const_cmp, const_index)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

macro_rules! all_eq {
    ($a: expr, $b: expr $(,)?) => {{
        let a = $a;
        let b = $b;
        assert!(a == b);
    }};
}

use std::intrinsics::simd::*;

const fn arithmetic() {
    let x1 = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
    let x2 = f32x4::from_array([2.0, 3.0, 4.0, 5.0]);

    unsafe {
        all_eq!(simd_add_algebraic(x1, x2), f32x4::from_array([3.0, 5.0, 7.0, 9.0]));
        all_eq!(simd_add_algebraic(x2, x1), f32x4::from_array([3.0, 5.0, 7.0, 9.0]));

        all_eq!(simd_mul_algebraic(x1, x2), f32x4::from_array([2.0, 6.0, 12.0, 20.0]));
        all_eq!(simd_mul_algebraic(x2, x1), f32x4::from_array([2.0, 6.0, 12.0, 20.0]));

        all_eq!(simd_sub_algebraic(x2, x1), f32x4::from_array([1.0, 1.0, 1.0, 1.0]));
        all_eq!(simd_sub_algebraic(x1, x2), f32x4::from_array([-1.0, -1.0, -1.0, -1.0]));

        all_eq!(
            simd_div_algebraic(x1, x2),
            f32x4::from_array([1.0 / 2.0, 2.0 / 3.0, 3.0 / 4.0, 4.0 / 5.0])
        );
        all_eq!(
            simd_div_algebraic(x2, x1),
            f32x4::from_array([2.0 / 1.0, 3.0 / 2.0, 4.0 / 3.0, 5.0 / 4.0])
        );

        all_eq!(simd_rem_algebraic(x1, x1), f32x4::from_array([0.0, 0.0, 0.0, 0.0]));
        all_eq!(simd_rem_algebraic(x1, x2), x1);
        all_eq!(simd_rem_algebraic(x2, x1), f32x4::from_array([0.0, 1.0, 1.0, 1.0]));

        assert_eq!(simd_reduce_add_algebraic(x1), 10.0_f32);
        assert_eq!(simd_reduce_add_algebraic(x2), 14.0_f32);
        assert_eq!(simd_reduce_mul_algebraic(x1), 24.0_f32);
        assert_eq!(simd_reduce_mul_algebraic(x2), 120.0_f32);
    }
}

fn main() {
    const { arithmetic() };
    arithmetic();
}
