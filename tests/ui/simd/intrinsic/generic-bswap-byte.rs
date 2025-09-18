//@ run-pass
#![feature(repr_simd, core_intrinsics, const_trait_impl, const_cmp, const_index)]

#[path = "../auxiliary/minisimd_const.rs"]
mod minisimd_const;
use minisimd_const::*;

use std::intrinsics::simd::simd_bswap;

make_runtime_and_compiletime! {
fn main() {
    unsafe {
        assert_eq_const_safe!(
            simd_bswap(i8x4::from_array([0, 1, 2, 3])).into_array(),
            [0, 1, 2, 3]
        );
        assert_eq_const_safe!(
            simd_bswap(u8x4::from_array([0, 1, 2, 3])).into_array(),
            [0, 1, 2, 3]
        );
    }
}
}
