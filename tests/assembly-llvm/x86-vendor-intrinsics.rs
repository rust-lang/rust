// Output differs depending on ABI so we need to match the full target.
//@ only-x86_64-unknown-linux-gnu
//@ assembly-output: emit-asm
//@ compile-flags: -Ctarget-feature=-sse3 -C opt-level=3

// Regression test for various cases where we used to compile x86 vendor intrinsics in a suboptimal
// way.

#![crate_type = "lib"]

use std::arch::x86_64::*;

// CHECK-LABEL: test_packus_epi16:
#[unsafe(no_mangle)]
#[target_feature(enable = "sse2")]
extern "C" fn test_packus_epi16(a: __m128i, b: __m128i) -> __m128i {
    // CHECK: .cfi_startproc
    // CHECK-NEXT: packuswb
    // CHECK-NEXT: ret
    _mm_packus_epi16(a, b)
}
