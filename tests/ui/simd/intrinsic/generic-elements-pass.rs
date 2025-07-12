//@ run-pass

#![feature(repr_simd, intrinsics, core_intrinsics)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{
    simd_extract, simd_extract_dyn, simd_insert, simd_insert_dyn, simd_shuffle,
};

#[repr(simd)]
struct SimdShuffleIdx<const LEN: usize>([u32; LEN]);

macro_rules! all_eq {
    ($a: expr, $b: expr) => {{
        let a = $a;
        let b = $b;
        // type inference works better with the concrete type on the
        // left, but humans work better with the expected on the
        // right.
        assert!(b == a, "{:?} != {:?}", a, b);
    }};
}

fn main() {
    let x2 = i32x2::from_array([20, 21]);
    let x4 = i32x4::from_array([40, 41, 42, 43]);
    let x8 = i32x8::from_array([80, 81, 82, 83, 84, 85, 86, 87]);
    unsafe {
        all_eq!(simd_insert(x2, 0, 100), i32x2::from_array([100, 21]));
        all_eq!(simd_insert(x2, 1, 100), i32x2::from_array([20, 100]));

        all_eq!(simd_insert(x4, 0, 100), i32x4::from_array([100, 41, 42, 43]));
        all_eq!(simd_insert(x4, 1, 100), i32x4::from_array([40, 100, 42, 43]));
        all_eq!(simd_insert(x4, 2, 100), i32x4::from_array([40, 41, 100, 43]));
        all_eq!(simd_insert(x4, 3, 100), i32x4::from_array([40, 41, 42, 100]));

        all_eq!(simd_insert(x8, 0, 100), i32x8::from_array([100, 81, 82, 83, 84, 85, 86, 87]));
        all_eq!(simd_insert(x8, 1, 100), i32x8::from_array([80, 100, 82, 83, 84, 85, 86, 87]));
        all_eq!(simd_insert(x8, 2, 100), i32x8::from_array([80, 81, 100, 83, 84, 85, 86, 87]));
        all_eq!(simd_insert(x8, 3, 100), i32x8::from_array([80, 81, 82, 100, 84, 85, 86, 87]));
        all_eq!(simd_insert(x8, 4, 100), i32x8::from_array([80, 81, 82, 83, 100, 85, 86, 87]));
        all_eq!(simd_insert(x8, 5, 100), i32x8::from_array([80, 81, 82, 83, 84, 100, 86, 87]));
        all_eq!(simd_insert(x8, 6, 100), i32x8::from_array([80, 81, 82, 83, 84, 85, 100, 87]));
        all_eq!(simd_insert(x8, 7, 100), i32x8::from_array([80, 81, 82, 83, 84, 85, 86, 100]));

        all_eq!(simd_extract(x2, 0), 20);
        all_eq!(simd_extract(x2, 1), 21);

        all_eq!(simd_extract(x4, 0), 40);
        all_eq!(simd_extract(x4, 1), 41);
        all_eq!(simd_extract(x4, 2), 42);
        all_eq!(simd_extract(x4, 3), 43);

        all_eq!(simd_extract(x8, 0), 80);
        all_eq!(simd_extract(x8, 1), 81);
        all_eq!(simd_extract(x8, 2), 82);
        all_eq!(simd_extract(x8, 3), 83);
        all_eq!(simd_extract(x8, 4), 84);
        all_eq!(simd_extract(x8, 5), 85);
        all_eq!(simd_extract(x8, 6), 86);
        all_eq!(simd_extract(x8, 7), 87);
    }
    unsafe {
        all_eq!(simd_insert_dyn(x2, 0, 100), i32x2::from_array([100, 21]));
        all_eq!(simd_insert_dyn(x2, 1, 100), i32x2::from_array([20, 100]));

        all_eq!(simd_insert_dyn(x4, 0, 100), i32x4::from_array([100, 41, 42, 43]));
        all_eq!(simd_insert_dyn(x4, 1, 100), i32x4::from_array([40, 100, 42, 43]));
        all_eq!(simd_insert_dyn(x4, 2, 100), i32x4::from_array([40, 41, 100, 43]));
        all_eq!(simd_insert_dyn(x4, 3, 100), i32x4::from_array([40, 41, 42, 100]));

        all_eq!(simd_insert_dyn(x8, 0, 100), i32x8::from_array([100, 81, 82, 83, 84, 85, 86, 87]));
        all_eq!(simd_insert_dyn(x8, 1, 100), i32x8::from_array([80, 100, 82, 83, 84, 85, 86, 87]));
        all_eq!(simd_insert_dyn(x8, 2, 100), i32x8::from_array([80, 81, 100, 83, 84, 85, 86, 87]));
        all_eq!(simd_insert_dyn(x8, 3, 100), i32x8::from_array([80, 81, 82, 100, 84, 85, 86, 87]));
        all_eq!(simd_insert_dyn(x8, 4, 100), i32x8::from_array([80, 81, 82, 83, 100, 85, 86, 87]));
        all_eq!(simd_insert_dyn(x8, 5, 100), i32x8::from_array([80, 81, 82, 83, 84, 100, 86, 87]));
        all_eq!(simd_insert_dyn(x8, 6, 100), i32x8::from_array([80, 81, 82, 83, 84, 85, 100, 87]));
        all_eq!(simd_insert_dyn(x8, 7, 100), i32x8::from_array([80, 81, 82, 83, 84, 85, 86, 100]));

        all_eq!(simd_extract_dyn(x2, 0), 20);
        all_eq!(simd_extract_dyn(x2, 1), 21);

        all_eq!(simd_extract_dyn(x4, 0), 40);
        all_eq!(simd_extract_dyn(x4, 1), 41);
        all_eq!(simd_extract_dyn(x4, 2), 42);
        all_eq!(simd_extract_dyn(x4, 3), 43);

        all_eq!(simd_extract_dyn(x8, 0), 80);
        all_eq!(simd_extract_dyn(x8, 1), 81);
        all_eq!(simd_extract_dyn(x8, 2), 82);
        all_eq!(simd_extract_dyn(x8, 3), 83);
        all_eq!(simd_extract_dyn(x8, 4), 84);
        all_eq!(simd_extract_dyn(x8, 5), 85);
        all_eq!(simd_extract_dyn(x8, 6), 86);
        all_eq!(simd_extract_dyn(x8, 7), 87);
    }

    let y2 = i32x2::from_array([120, 121]);
    let y4 = i32x4::from_array([140, 141, 142, 143]);
    let y8 = i32x8::from_array([180, 181, 182, 183, 184, 185, 186, 187]);
    unsafe {
        all_eq!(
            simd_shuffle(x2, y2, const { SimdShuffleIdx([3u32, 0]) }),
            i32x2::from_array([121, 20])
        );
        all_eq!(
            simd_shuffle(x2, y2, const { SimdShuffleIdx([3u32, 0, 1, 2]) }),
            i32x4::from_array([121, 20, 21, 120])
        );
        all_eq!(
            simd_shuffle(x2, y2, const { SimdShuffleIdx([3u32, 0, 1, 2, 1, 2, 3, 0]) }),
            i32x8::from_array([121, 20, 21, 120, 21, 120, 121, 20])
        );

        all_eq!(
            simd_shuffle(x4, y4, const { SimdShuffleIdx([7u32, 2]) }),
            i32x2::from_array([143, 42])
        );
        all_eq!(
            simd_shuffle(x4, y4, const { SimdShuffleIdx([7u32, 2, 5, 0]) }),
            i32x4::from_array([143, 42, 141, 40])
        );
        all_eq!(
            simd_shuffle(x4, y4, const { SimdShuffleIdx([7u32, 2, 5, 0, 3, 6, 4, 1]) }),
            i32x8::from_array([143, 42, 141, 40, 43, 142, 140, 41])
        );

        all_eq!(
            simd_shuffle(x8, y8, const { SimdShuffleIdx([11u32, 5]) }),
            i32x2::from_array([183, 85])
        );
        all_eq!(
            simd_shuffle(x8, y8, const { SimdShuffleIdx([11u32, 5, 15, 0]) }),
            i32x4::from_array([183, 85, 187, 80])
        );
        all_eq!(
            simd_shuffle(x8, y8, const { SimdShuffleIdx([11u32, 5, 15, 0, 3, 8, 12, 1]) }),
            i32x8::from_array([183, 85, 187, 80, 83, 180, 184, 81])
        );
    }
}
