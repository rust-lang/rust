#![cfg(not(test))]

// These symbols are all defined by `libm`,
// or by `compiler-builtins` on unsupported platforms.
extern "C" {
    pub fn acos(n: f64) -> f64;
    pub fn asin(n: f64) -> f64;
    pub fn atan(n: f64) -> f64;
    pub fn atan2(a: f64, b: f64) -> f64;
    pub fn cbrt(n: f64) -> f64;
    pub fn cbrtf(n: f32) -> f32;
    pub fn cosh(n: f64) -> f64;
    pub fn expm1(n: f64) -> f64;
    pub fn expm1f(n: f32) -> f32;
    pub fn fdim(a: f64, b: f64) -> f64;
    pub fn fdimf(a: f32, b: f32) -> f32;
    #[cfg_attr(target_env = "msvc", link_name = "_hypot")]
    pub fn hypot(x: f64, y: f64) -> f64;
    #[cfg_attr(target_env = "msvc", link_name = "_hypotf")]
    pub fn hypotf(x: f32, y: f32) -> f32;
    pub fn log1p(n: f64) -> f64;
    pub fn log1pf(n: f32) -> f32;
    pub fn sinh(n: f64) -> f64;
    pub fn tan(n: f64) -> f64;
    pub fn tanh(n: f64) -> f64;
    pub fn tgamma(n: f64) -> f64;
    pub fn tgammaf(n: f32) -> f32;
    pub fn lgamma_r(n: f64, s: &mut i32) -> f64;
    #[cfg(not(target_os = "aix"))]
    pub fn lgammaf_r(n: f32, s: &mut i32) -> f32;

    pub fn acosf128(n: f128) -> f128;
    pub fn asinf128(n: f128) -> f128;
    pub fn atanf128(n: f128) -> f128;
    pub fn atan2f128(a: f128, b: f128) -> f128;
    pub fn cbrtf128(n: f128) -> f128;
    pub fn coshf128(n: f128) -> f128;
    pub fn expm1f128(n: f128) -> f128;
    pub fn hypotf128(x: f128, y: f128) -> f128;
    pub fn log1pf128(n: f128) -> f128;
    pub fn sinhf128(n: f128) -> f128;
    pub fn tanf128(n: f128) -> f128;
    pub fn tanhf128(n: f128) -> f128;
    pub fn tgammaf128(n: f128) -> f128;
    pub fn lgammaf128_r(n: f128, s: &mut i32) -> f128;

    cfg_if::cfg_if! {
    if #[cfg(not(all(target_os = "windows", target_env = "msvc", target_arch = "x86")))] {
        pub fn acosf(n: f32) -> f32;
        pub fn asinf(n: f32) -> f32;
        pub fn atan2f(a: f32, b: f32) -> f32;
        pub fn atanf(n: f32) -> f32;
        pub fn coshf(n: f32) -> f32;
        pub fn sinhf(n: f32) -> f32;
        pub fn tanf(n: f32) -> f32;
        pub fn tanhf(n: f32) -> f32;
    }}
}

// On AIX, we don't have lgammaf_r only the f64 version, so we can
// use the f64 version lgamma_r
#[cfg(target_os = "aix")]
pub unsafe fn lgammaf_r(n: f32, s: &mut i32) -> f32 {
    lgamma_r(n.into(), s) as f32
}

// On 32-bit x86 MSVC these functions aren't defined, so we just define shims
// which promote everything to f64, perform the calculation, and then demote
// back to f32. While not precisely correct should be "correct enough" for now.
cfg_if::cfg_if! {
if #[cfg(all(target_os = "windows", target_env = "msvc", target_arch = "x86"))] {
    #[inline]
   pub unsafe fn acosf(n: f32) -> f32 {
        f64::acos(n as f64) as f32
    }

    #[inline]
    pub unsafe fn asinf(n: f32) -> f32 {
        f64::asin(n as f64) as f32
    }

    #[inline]
    pub unsafe fn atan2f(n: f32, b: f32) -> f32 {
        f64::atan2(n as f64, b as f64) as f32
    }

    #[inline]
    pub unsafe fn atanf(n: f32) -> f32 {
        f64::atan(n as f64) as f32
    }

    #[inline]
    pub unsafe fn coshf(n: f32) -> f32 {
        f64::cosh(n as f64) as f32
    }

    #[inline]
    pub unsafe fn sinhf(n: f32) -> f32 {
        f64::sinh(n as f64) as f32
    }

    #[inline]
    pub unsafe fn tanf(n: f32) -> f32 {
        f64::tan(n as f64) as f32
    }

    #[inline]
    pub unsafe fn tanhf(n: f32) -> f32 {
        f64::tanh(n as f64) as f32
    }
}}
