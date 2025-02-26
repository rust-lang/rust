//@ only-x86_64
// Set the base cpu explicitly, in case the default has been changed.
//@ compile-flags: -Copt-level=3 -Ctarget-cpu=x86-64

#![crate_type = "lib"]

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

// Don't allow the above CHECK-NOT to accidentally match a commit hash in the
// compiler version.
// CHECK-LABEL: rustc version
