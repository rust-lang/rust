//@ only-x86_64
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

// See <https://github.com/rust-lang/rust/issues/159831> for context.
// CHECK-LABEL: shlv_u16x16:
#[unsafe(no_mangle)]
#[target_feature(enable = "avx2")]
extern "C" fn shlv_u16x16(a: __m256i, count: __m256i) -> __m256i {
    // CHECK: .cfi_startproc
    // CHECK-NOT: vpcmp
    // CHECK-NOT: vmov
    // CHECK: vpsllvw
    // CHECK-NOT: vpcmp
    // CHECK-NOT: vmov
    // CHECK: ret
    let low_words = _mm256_set1_epi32(0x0000_ffff);
    let low_count = _mm256_and_si256(count, low_words);
    let high_count = _mm256_srli_epi32::<16>(count);

    let high_values = _mm256_andnot_si256(low_words, a);
    let low_shifted = _mm256_sllv_epi32(a, low_count);
    let high_shifted = _mm256_sllv_epi32(high_values, high_count);
    let low_shifted = _mm256_and_si256(low_shifted, low_words);

    _mm256_or_si256(low_shifted, high_shifted)
}

// See <https://github.com/rust-lang/rust/issues/159831> for context.
// CHECK-LABEL: shrv_u16x16:
#[unsafe(no_mangle)]
#[target_feature(enable = "avx2")]
extern "C" fn shrv_u16x16(a: __m256i, count: __m256i) -> __m256i {
    // CHECK: .cfi_startproc
    // CHECK-NOT: vpcmp
    // CHECK-NOT: vmov
    // CHECK: vpsrlvw
    // CHECK-NOT: vpcmp
    // CHECK-NOT: vmov
    // CHECK: ret
    let low_words = _mm256_set1_epi32(0x0000_ffff);
    let low_count = _mm256_and_si256(count, low_words);
    let high_count = _mm256_srli_epi32::<16>(count);

    let low_values = _mm256_and_si256(a, low_words);
    let low_shifted = _mm256_srlv_epi32(low_values, low_count);
    let high_shifted = _mm256_srlv_epi32(a, high_count);
    let high_shifted = _mm256_andnot_si256(low_words, high_shifted);

    _mm256_or_si256(low_shifted, high_shifted)
}

// See <https://github.com/rust-lang/rust/issues/159801> for context.
// CHECK-LABEL: test_sllv_srlv:
#[unsafe(no_mangle)]
#[target_feature(enable = "avx512bw")]
extern "C" fn test_sllv_srlv(win: __m512i, x: __m512i, v: __m512i) -> __m512i {
    // CHECK: .cfi_startproc
    // CHECK-NEXT: vpsllvw
    // CHECK-NEXT: vporq
    // CHECK-NEXT: vpandd
    // CHECK-NEXT: vpsrlvw
    // CHECK-NEXT: ret
    let w = _mm512_or_si512(win, _mm512_sllv_epi16(x, v));
    _mm512_srlv_epi16(w, _mm512_and_si512(w, _mm512_set1_epi16(7)))
}
