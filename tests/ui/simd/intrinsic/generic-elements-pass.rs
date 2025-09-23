//@ run-pass
//@ compile-flags: --cfg minisimd_const

#![feature(repr_simd, core_intrinsics, const_trait_impl, const_cmp, const_index)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{
    simd_extract, simd_extract_dyn, simd_insert, simd_insert_dyn, simd_shuffle,
};

#[repr(simd)]
struct SimdShuffleIdx<const LEN: usize>([u32; LEN]);

fn extract_insert_dyn() {
    let x2 = i32x2::from_array([20, 21]);
    let x4 = i32x4::from_array([40, 41, 42, 43]);
    let x8 = i32x8::from_array([80, 81, 82, 83, 84, 85, 86, 87]);

    unsafe {
        assert_eq_const_safe!(simd_insert_dyn(x2, 0, 100), i32x2::from_array([100, 21]));
        assert_eq_const_safe!(simd_insert_dyn(x2, 1, 100), i32x2::from_array([20, 100]));

        assert_eq_const_safe!(simd_insert_dyn(x4, 0, 100), i32x4::from_array([100, 41, 42, 43]));
        assert_eq_const_safe!(simd_insert_dyn(x4, 1, 100), i32x4::from_array([40, 100, 42, 43]));
        assert_eq_const_safe!(simd_insert_dyn(x4, 2, 100), i32x4::from_array([40, 41, 100, 43]));
        assert_eq_const_safe!(simd_insert_dyn(x4, 3, 100), i32x4::from_array([40, 41, 42, 100]));

        assert_eq_const_safe!(
            simd_insert_dyn(x8, 0, 100),
            i32x8::from_array([100, 81, 82, 83, 84, 85, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert_dyn(x8, 1, 100),
            i32x8::from_array([80, 100, 82, 83, 84, 85, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert_dyn(x8, 2, 100),
            i32x8::from_array([80, 81, 100, 83, 84, 85, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert_dyn(x8, 3, 100),
            i32x8::from_array([80, 81, 82, 100, 84, 85, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert_dyn(x8, 4, 100),
            i32x8::from_array([80, 81, 82, 83, 100, 85, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert_dyn(x8, 5, 100),
            i32x8::from_array([80, 81, 82, 83, 84, 100, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert_dyn(x8, 6, 100),
            i32x8::from_array([80, 81, 82, 83, 84, 85, 100, 87])
        );
        assert_eq_const_safe!(
            simd_insert_dyn(x8, 7, 100),
            i32x8::from_array([80, 81, 82, 83, 84, 85, 86, 100])
        );

        assert_eq_const_safe!(simd_extract_dyn(x2, 0), 20);
        assert_eq_const_safe!(simd_extract_dyn(x2, 1), 21);

        assert_eq_const_safe!(simd_extract_dyn(x4, 0), 40);
        assert_eq_const_safe!(simd_extract_dyn(x4, 1), 41);
        assert_eq_const_safe!(simd_extract_dyn(x4, 2), 42);
        assert_eq_const_safe!(simd_extract_dyn(x4, 3), 43);

        assert_eq_const_safe!(simd_extract_dyn(x8, 0), 80);
        assert_eq_const_safe!(simd_extract_dyn(x8, 1), 81);
        assert_eq_const_safe!(simd_extract_dyn(x8, 2), 82);
        assert_eq_const_safe!(simd_extract_dyn(x8, 3), 83);
        assert_eq_const_safe!(simd_extract_dyn(x8, 4), 84);
        assert_eq_const_safe!(simd_extract_dyn(x8, 5), 85);
        assert_eq_const_safe!(simd_extract_dyn(x8, 6), 86);
        assert_eq_const_safe!(simd_extract_dyn(x8, 7), 87);
    }
}

macro_rules! simd_shuffle {
    ($a:expr, $b:expr, $swizzle:expr) => {
        simd_shuffle($a, $b, const { SimdShuffleIdx($swizzle) })
    };
}

const fn swizzle() {
    let x2 = i32x2::from_array([20, 21]);
    let x4 = i32x4::from_array([40, 41, 42, 43]);
    let x8 = i32x8::from_array([80, 81, 82, 83, 84, 85, 86, 87]);
    unsafe {
        assert_eq_const_safe!(simd_insert(x2, 0, 100), i32x2::from_array([100, 21]));
        assert_eq_const_safe!(simd_insert(x2, 1, 100), i32x2::from_array([20, 100]));

        assert_eq_const_safe!(simd_insert(x4, 0, 100), i32x4::from_array([100, 41, 42, 43]));
        assert_eq_const_safe!(simd_insert(x4, 1, 100), i32x4::from_array([40, 100, 42, 43]));
        assert_eq_const_safe!(simd_insert(x4, 2, 100), i32x4::from_array([40, 41, 100, 43]));
        assert_eq_const_safe!(simd_insert(x4, 3, 100), i32x4::from_array([40, 41, 42, 100]));

        assert_eq_const_safe!(
            simd_insert(x8, 0, 100),
            i32x8::from_array([100, 81, 82, 83, 84, 85, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert(x8, 1, 100),
            i32x8::from_array([80, 100, 82, 83, 84, 85, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert(x8, 2, 100),
            i32x8::from_array([80, 81, 100, 83, 84, 85, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert(x8, 3, 100),
            i32x8::from_array([80, 81, 82, 100, 84, 85, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert(x8, 4, 100),
            i32x8::from_array([80, 81, 82, 83, 100, 85, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert(x8, 5, 100),
            i32x8::from_array([80, 81, 82, 83, 84, 100, 86, 87])
        );
        assert_eq_const_safe!(
            simd_insert(x8, 6, 100),
            i32x8::from_array([80, 81, 82, 83, 84, 85, 100, 87])
        );
        assert_eq_const_safe!(
            simd_insert(x8, 7, 100),
            i32x8::from_array([80, 81, 82, 83, 84, 85, 86, 100])
        );

        assert_eq_const_safe!(simd_extract(x2, 0), 20);
        assert_eq_const_safe!(simd_extract(x2, 1), 21);

        assert_eq_const_safe!(simd_extract(x4, 0), 40);
        assert_eq_const_safe!(simd_extract(x4, 1), 41);
        assert_eq_const_safe!(simd_extract(x4, 2), 42);
        assert_eq_const_safe!(simd_extract(x4, 3), 43);

        assert_eq_const_safe!(simd_extract(x8, 0), 80);
        assert_eq_const_safe!(simd_extract(x8, 1), 81);
        assert_eq_const_safe!(simd_extract(x8, 2), 82);
        assert_eq_const_safe!(simd_extract(x8, 3), 83);
        assert_eq_const_safe!(simd_extract(x8, 4), 84);
        assert_eq_const_safe!(simd_extract(x8, 5), 85);
        assert_eq_const_safe!(simd_extract(x8, 6), 86);
        assert_eq_const_safe!(simd_extract(x8, 7), 87);
    }

    let y2 = i32x2::from_array([120, 121]);
    let y4 = i32x4::from_array([140, 141, 142, 143]);
    let y8 = i32x8::from_array([180, 181, 182, 183, 184, 185, 186, 187]);
    unsafe {
        assert_eq_const_safe!(simd_shuffle!(x2, y2, [3u32, 0]), i32x2::from_array([121, 20]));
        assert_eq_const_safe!(
            simd_shuffle!(x2, y2, [3u32, 0, 1, 2]),
            i32x4::from_array([121, 20, 21, 120])
        );
        assert_eq_const_safe!(
            simd_shuffle!(x2, y2, [3u32, 0, 1, 2, 1, 2, 3, 0]),
            i32x8::from_array([121, 20, 21, 120, 21, 120, 121, 20])
        );

        assert_eq_const_safe!(simd_shuffle!(x4, y4, [7u32, 2]), i32x2::from_array([143, 42]));
        assert_eq_const_safe!(
            simd_shuffle!(x4, y4, [7u32, 2, 5, 0]),
            i32x4::from_array([143, 42, 141, 40])
        );
        assert_eq_const_safe!(
            simd_shuffle!(x4, y4, [7u32, 2, 5, 0, 3, 6, 4, 1]),
            i32x8::from_array([143, 42, 141, 40, 43, 142, 140, 41])
        );

        assert_eq_const_safe!(simd_shuffle!(x8, y8, [11u32, 5]), i32x2::from_array([183, 85]));
        assert_eq_const_safe!(
            simd_shuffle!(x8, y8, [11u32, 5, 15, 0]),
            i32x4::from_array([183, 85, 187, 80])
        );
        assert_eq_const_safe!(
            simd_shuffle!(x8, y8, [11u32, 5, 15, 0, 3, 8, 12, 1]),
            i32x8::from_array([183, 85, 187, 80, 83, 180, 184, 81])
        );
    }
}

fn main() {
    extract_insert_dyn();
    const { swizzle() };
    swizzle();
}
