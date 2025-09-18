//@ run-pass
#![feature(repr_simd, core_intrinsics, const_trait_impl, const_cmp, const_index)]

#[path = "../auxiliary/minisimd_const.rs"]
mod minisimd_const;
use minisimd_const::*;

use std::intrinsics::simd::simd_cast;

type V<T> = Simd<T, 4>;

make_runtime_and_compiletime! {
    fn main() {
        let u: V::<usize> = Simd([0, 1, 2, 3]);
        let uu32: V<u32> = unsafe { simd_cast(u) };
        let ui64: V<i64> = unsafe { simd_cast(u) };

        assert_eq_const_safe!(uu32, V::<u32>::from_array([0, 1, 2, 3]));
        assert_eq_const_safe!(ui64, V::<i64>::from_array([0, 1, 2, 3]));
    }
}
