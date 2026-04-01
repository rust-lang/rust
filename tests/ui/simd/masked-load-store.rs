//@ ignore-backends: gcc
//@ compile-flags: --cfg minisimd_const
//@ run-pass
#![feature(repr_simd, core_intrinsics, const_trait_impl, const_cmp, const_index)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{SimdAlign, simd_masked_load, simd_masked_store};

const fn masked_load_store() {
    unsafe {
        let a = Simd::<u8, 4>([0, 1, 2, 3]);
        let b_src = [4u8, 5, 6, 7];
        let b_default = Simd::<u8, 4>([9; 4]);
        let b: Simd<u8, 4> = simd_masked_load::<_, _, _, { SimdAlign::Element }>(
            Simd::<i8, 4>([-1, 0, -1, -1]),
            b_src.as_ptr(),
            b_default,
        );

        assert_eq!(b.as_array(), &[4, 9, 6, 7]);

        let mut output = [u8::MAX; 5];

        simd_masked_store::<_, _, _, { SimdAlign::Element }>(
            Simd::<i8, 4>([-1, -1, -1, 0]),
            output.as_mut_ptr(),
            a,
        );
        assert_eq!(&output, &[0, 1, 2, u8::MAX, u8::MAX]);
        simd_masked_store::<_, _, _, { SimdAlign::Element }>(
            Simd::<i8, 4>([0, -1, -1, 0]),
            output[1..].as_mut_ptr(),
            b,
        );
        assert_eq!(&output, &[0, 1, 9, 6, u8::MAX]);
    }
}

fn main() {
    const { masked_load_store() };
    masked_load_store();
}
