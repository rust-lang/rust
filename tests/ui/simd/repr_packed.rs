//@ run-pass

#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_add;

fn check_size_align<T, const N: usize>() {
    use std::mem;
    assert_eq!(mem::size_of::<PackedSimd<T, N>>(), mem::size_of::<[T; N]>());
    assert_eq!(mem::size_of::<PackedSimd<T, N>>() % mem::align_of::<PackedSimd<T, N>>(), 0);
}

fn check_ty<T>() {
    check_size_align::<T, 1>();
    check_size_align::<T, 2>();
    check_size_align::<T, 3>();
    check_size_align::<T, 4>();
    check_size_align::<T, 8>();
    check_size_align::<T, 9>();
    check_size_align::<T, 15>();
}

fn main() {
    check_ty::<u8>();
    check_ty::<i16>();
    check_ty::<u32>();
    check_ty::<i64>();
    check_ty::<usize>();
    check_ty::<f32>();
    check_ty::<f64>();

    unsafe {
        // powers-of-two have no padding and have the same layout as #[repr(simd)]
        let x: PackedSimd<f64, 4> =
            simd_add(
                PackedSimd::<f64, 4>([0., 1., 2., 3.]),
                PackedSimd::<f64, 4>([2., 2., 2., 2.]),
            );
        assert_eq!(x.into_array(), [2., 3., 4., 5.]);

        // non-powers-of-two should have padding (which is removed by #[repr(packed)]),
        // but the intrinsic handles it
        let x: PackedSimd<f64, 3> =
            simd_add(
                PackedSimd::<f64, 3>([0., 1., 2.]),
                PackedSimd::<f64, 3>([2., 2., 2.]),
            );
        let arr: [f64; 3] = x.into_array();
        assert_eq!(arr, [2., 3., 4.]);
    }
}
