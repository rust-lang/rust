// This used to cause an ICE for an internal index out of range due to simd_shuffle_indices being
// passed the wrong Instance, causing issues with inlining. See #67557.
//
//@ run-pass
//@ compile-flags: -Zmir-opt-level=4
#![feature(intrinsics, repr_simd)]

extern "rust-intrinsic" {
    fn simd_shuffle<T, I, U>(x: T, y: T, idx: I) -> U;
}

#[repr(simd)]
#[derive(Debug, PartialEq)]
struct Simd2(u8, u8);

fn main() {
    unsafe {
        let _: Simd2 = inline_me();
    }
}

#[inline(always)]
unsafe fn inline_me() -> Simd2 {
    const IDX: [u32; 2] = [0, 3];
    simd_shuffle(Simd2(10, 11), Simd2(12, 13), IDX)
}
