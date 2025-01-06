//! Architecture-specific support for x86-32 and x86-64 with SSE2

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub fn sqrtf(x: f32) -> f32 {
    unsafe {
        let m = _mm_set_ss(x);
        let m_sqrt = _mm_sqrt_ss(m);
        _mm_cvtss_f32(m_sqrt)
    }
}

pub fn sqrt(x: f64) -> f64 {
    unsafe {
        let m = _mm_set_sd(x);
        let m_sqrt = _mm_sqrt_pd(m);
        _mm_cvtsd_f64(m_sqrt)
    }
}
