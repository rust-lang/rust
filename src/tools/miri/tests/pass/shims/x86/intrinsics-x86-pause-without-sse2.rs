// Ignore everything except x86 and x86_64
// Any new targets that are added to CI should be ignored here.
// (We cannot use `cfg`-based tricks here since the `target-feature` flags below only work on x86.)
//@ignore-target-aarch64
//@ignore-target-arm
//@ignore-target-avr
//@ignore-target-s390x
//@ignore-target-thumbv7em
//@ignore-target-wasm32
//@compile-flags: -C target-feature=-sse2

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    assert!(!is_x86_feature_detected!("sse2"));

    unsafe {
        // This is a SSE2 intrinsic, but it behaves as a no-op when SSE2
        // is not available, so it is always safe to call.
        _mm_pause();
    }
}
