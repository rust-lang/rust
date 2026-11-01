// This used to cause an ICE for an internal index out of range due to simd_shuffle_indices being
// passed the wrong Instance, causing issues with inlining. See #67557.
//
//@ run-pass
//@ compile-flags: -Zmir-opt-level=4
#![feature(core_intrinsics, repr_simd)]

use std::intrinsics::simd::simd_shuffle;

#[repr(simd)]
#[derive(Debug, PartialEq)]
struct Simd2([u8; 2]);

#[repr(simd)]
struct SimdShuffleIdx<const LEN: usize>([u32; LEN]);

fn main() {
    unsafe {
        let _: Simd2 = inline_me();
    }
}

#[inline(always)]
unsafe fn inline_me() -> Simd2 {
    const IDX: SimdShuffleIdx<2> = SimdShuffleIdx([0, 3]);
    simd_shuffle(Simd2([10, 11]), Simd2([12, 13]), IDX)
}
