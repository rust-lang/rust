// Ignore everything except x86 and x86_64
// Any new targets that are added to CI should be ignored here.
// We cannot use `cfg`-based tricks here since the output would be
// different for non-x86 targets.
//@only-target: x86_64 i686
// Explicitly disable SSE4.1 because it is enabled by default on macOS
//@compile-flags: -C target-feature=-sse4.1

#![feature(link_llvm_intrinsics, simd_ffi)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    assert!(is_x86_feature_detected!("sse"));
    assert!(!is_x86_feature_detected!("sse4.1"));

    unsafe {
        // Pass, since SSE is enabled
        minss(_mm_setzero_ps(), _mm_setzero_ps());

        // Fail, since SSE4.1 is not enabled
        dpps(_mm_setzero_ps(), _mm_setzero_ps(), 0);
        //~^ ERROR: Undefined Behavior: attempted to call intrinsic `llvm.x86.sse41.dpps` that requires missing target feature sse4.1
    }
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sse.min.ss"]
    fn minss(a: __m128, b: __m128) -> __m128;

    #[link_name = "llvm.x86.sse41.dpps"]
    fn dpps(a: __m128, b: __m128, imm8: u8) -> __m128;
}
