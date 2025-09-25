// We're testing x86 target specific features
//@only-target: x86_64 i686

//! rsqrt and rcp SSE/AVX operations are approximate. We use that as license to treat them as
//! non-deterministic. Ensure that we do indeed see random results within the expected error bounds.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::collections::HashSet;

fn main() {
    let mut vals = HashSet::new();
    for _ in 0..50 {
        unsafe {
            // Compute the inverse square root of 4.0, four times.
            let a = _mm_setr_ps(4.0, 4.0, 4.0, 4.0);
            let exact = 0.5;
            let r = _mm_rsqrt_ps(a);
            let r: [f32; 4] = std::mem::transmute(r);
            // Check the results.
            for r in r {
                vals.insert(r.to_bits());
                // Ensure the relative error is less than 2^-12.
                let rel_error = (r - exact) / exact;
                let log_error = rel_error.abs().log2();
                assert!(
                    rel_error == 0.0 || log_error < -12.0,
                    "got an error of {rel_error} = 2^{log_error}"
                );
            }
        }
    }
    // Ensure we saw a bunch of different results.
    assert!(vals.len() >= 50);
}
