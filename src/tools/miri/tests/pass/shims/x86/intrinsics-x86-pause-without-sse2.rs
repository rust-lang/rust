// We're testing x86-32 target specific features. SSE always exists on x86-64.
//@only-target: i686
//@compile-flags: -C target-feature=-sse2

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    assert!(!is_x86_feature_detected!("sse2"));
    _mm_pause();
}
