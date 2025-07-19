// This used to cause assert_10_13 to unexpectingly fail, due to simd_shuffle_indices being passed
// the wrong Instance, causing issues with inlining. See #67557.
//
//@ run-pass
//@ compile-flags: -Zmir-opt-level=4
#![feature(core_intrinsics, repr_simd)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_shuffle;

type Simd2 = u8x2;

#[repr(simd)]
struct SimdShuffleIdx<const LEN: usize>([u32; LEN]);

fn main() {
    unsafe {
        const IDX: SimdShuffleIdx<2> = SimdShuffleIdx([0, 1]);
        let p_res: Simd2 = simd_shuffle(
            Simd2::from_array([10, 11]),
            Simd2::from_array([12, 13]),
            IDX,
        );
        let a_res: Simd2 = inline_me();

        assert_10_11(p_res);
        assert_10_13(a_res);
    }
}

#[inline(never)]
fn assert_10_11(x: Simd2) {
    assert_eq!(x.into_array(), [10, 11]);
}

#[inline(never)]
fn assert_10_13(x: Simd2) {
    assert_eq!(x.into_array(), [10, 13]);
}

#[inline(always)]
unsafe fn inline_me() -> Simd2 {
    const IDX: SimdShuffleIdx<2> = SimdShuffleIdx([0, 3]);
    simd_shuffle(Simd2::from_array([10, 11]), Simd2::from_array([12, 13]), IDX)
}
