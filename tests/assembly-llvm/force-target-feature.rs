//@ only-x86_64
//@ assembly-output: emit-asm
//@ compile-flags: -C opt-level=3 -C target-feature=-avx2
//@ ignore-sgx Tests incompatible with LVI mitigations

#![feature(effective_target_features)]

use std::arch::x86_64::{__m256i, _mm256_add_epi32, _mm256_setzero_si256};
use std::ops::Add;

#[derive(Clone, Copy)]
struct AvxU32(__m256i);

impl Add<AvxU32> for AvxU32 {
    type Output = Self;

    #[no_mangle]
    #[inline(never)]
    #[unsafe(force_target_feature(enable = "avx2"))]
    fn add(self, oth: AvxU32) -> AvxU32 {
        // CHECK-LABEL: add:
        // CHECK-NOT: callq
        // CHECK: vpaddd
        // CHECK: retq
        Self(_mm256_add_epi32(self.0, oth.0))
    }
}

fn main() {
    assert!(is_x86_feature_detected!("avx2"));
    let v = AvxU32(unsafe { _mm256_setzero_si256() });
    v + v;
}
