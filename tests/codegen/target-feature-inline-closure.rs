// only-x86_64
// compile-flags: -Copt-level=3

#![crate_type = "lib"]
#![feature(target_feature_11)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// CHECK-LABEL: @with_avx
#[no_mangle]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn with_avx(x: __m256) -> __m256 {
    // CHECK: fadd
    let add = {
        #[inline(always)]
        |x, y| unsafe { _mm256_add_ps(x, y) }
    };
    add(x, x)
}

// CHECK-LABEL: @without_avx
#[no_mangle]
#[cfg(target_arch = "x86_64")]
unsafe fn without_avx(x: __m256) -> __m256 {
    // CHECK-NOT: fadd
    let add = {
        #[inline(always)]
        |x, y| unsafe { _mm256_add_ps(x, y) }
    };
    add(x, x)
}
