//! Bindings to math functions provided by the system `libm` or by the `libm` crate, exposed
//! via `compiler-builtins`.
//!
//! The functions in the root of this module are "guaranteed" to be available; see the
//! `full_availability` module in compiler-builtins for details.

// SAFETY: These symbols have standard interfaces in C and are defined by `libm`, or are
// provided by `compiler-builtins` on unsupported platforms.
#[allow(dead_code)] // This list reflects what is available rather than what is consumed.
unsafe extern "C" {
    pub(crate) safe fn cbrt(x: f64) -> f64;
    pub(crate) safe fn cbrtf(n: f32) -> f32;
    pub(crate) safe fn ceil(x: f64) -> f64;
    pub(crate) safe fn ceilf(x: f32) -> f32;
    pub(crate) safe fn ceilf128(x: f128) -> f128;
    pub(crate) safe fn ceilf16(x: f16) -> f16;
    pub(crate) safe fn copysign(x: f64, y: f64) -> f64;
    pub(crate) safe fn copysignf(x: f32, y: f32) -> f32;
    pub(crate) safe fn copysignf128(x: f128, y: f128) -> f128;
    pub(crate) safe fn copysignf16(x: f16, y: f16) -> f16;
    pub(crate) safe fn fabs(x: f64) -> f64;
    pub(crate) safe fn fabsf(x: f32) -> f32;
    pub(crate) safe fn fabsf128(x: f128) -> f128;
    pub(crate) safe fn fabsf16(x: f16) -> f16;
    pub(crate) safe fn fdim(a: f64, b: f64) -> f64;
    pub(crate) safe fn fdimf(a: f32, b: f32) -> f32;
    pub(crate) safe fn fdimf128(x: f128, y: f128) -> f128;
    pub(crate) safe fn fdimf16(x: f16, y: f16) -> f16;
    pub(crate) safe fn floor(x: f64) -> f64;
    pub(crate) safe fn floorf(x: f32) -> f32;
    pub(crate) safe fn floorf128(x: f128) -> f128;
    pub(crate) safe fn floorf16(x: f16) -> f16;
    pub(crate) safe fn fma(x: f64, y: f64, z: f64) -> f64;
    pub(crate) safe fn fmaf(x: f32, y: f32, z: f32) -> f32;
    pub(crate) safe fn fmaf128(x: f128, y: f128, z: f128) -> f128;
    pub(crate) safe fn fmax(x: f64, y: f64) -> f64;
    pub(crate) safe fn fmaxf(x: f32, y: f32) -> f32;
    pub(crate) safe fn fmaxf128(x: f128, y: f128) -> f128;
    pub(crate) safe fn fmaxf16(x: f16, y: f16) -> f16;
    pub(crate) safe fn fmaximum(x: f64, y: f64) -> f64;
    pub(crate) safe fn fmaximumf(x: f32, y: f32) -> f32;
    pub(crate) safe fn fmaximumf128(x: f128, y: f128) -> f128;
    pub(crate) safe fn fmaximumf16(x: f16, y: f16) -> f16;
    pub(crate) safe fn fmin(x: f64, y: f64) -> f64;
    pub(crate) safe fn fminf(x: f32, y: f32) -> f32;
    pub(crate) safe fn fminf128(x: f128, y: f128) -> f128;
    pub(crate) safe fn fminf16(x: f16, y: f16) -> f16;
    pub(crate) safe fn fminimum(x: f64, y: f64) -> f64;
    pub(crate) safe fn fminimumf(x: f32, y: f32) -> f32;
    pub(crate) safe fn fminimumf128(x: f128, y: f128) -> f128;
    pub(crate) safe fn fminimumf16(x: f16, y: f16) -> f16;
    pub(crate) safe fn fmod(x: f64, y: f64) -> f64;
    pub(crate) safe fn fmodf(x: f32, y: f32) -> f32;
    pub(crate) safe fn fmodf128(x: f128, y: f128) -> f128;
    pub(crate) safe fn fmodf16(x: f16, y: f16) -> f16;
    pub(crate) safe fn rint(x: f64) -> f64;
    pub(crate) safe fn rintf(x: f32) -> f32;
    pub(crate) safe fn rintf128(x: f128) -> f128;
    pub(crate) safe fn rintf16(x: f16) -> f16;
    pub(crate) safe fn round(x: f64) -> f64;
    pub(crate) safe fn roundeven(x: f64) -> f64;
    pub(crate) safe fn roundevenf(x: f32) -> f32;
    pub(crate) safe fn roundevenf128(x: f128) -> f128;
    pub(crate) safe fn roundevenf16(x: f16) -> f16;
    pub(crate) safe fn roundf(x: f32) -> f32;
    pub(crate) safe fn roundf128(x: f128) -> f128;
    pub(crate) safe fn roundf16(x: f16) -> f16;
    pub(crate) safe fn sqrt(x: f64) -> f64;
    pub(crate) safe fn sqrtf(x: f32) -> f32;
    pub(crate) safe fn sqrtf128(x: f128) -> f128;
    pub(crate) safe fn sqrtf16(x: f16) -> f16;
    pub(crate) safe fn trunc(x: f64) -> f64;
    pub(crate) safe fn truncf(x: f32) -> f32;
    pub(crate) safe fn truncf128(x: f128) -> f128;
    pub(crate) safe fn truncf16(x: f16) -> f16;
}

/// These symbols will be available when `std` is available, and on many no-std platforms. However,
/// since this isn't a guarantee, we cannot rely on them for stable implementations.
pub(crate) mod likely_available {
    #[allow(dead_code)]
    unsafe extern "C" {
        pub(crate) safe fn acos(x: f64) -> f64;
        pub(crate) safe fn acosf(n: f32) -> f32;
        pub(crate) safe fn asin(x: f64) -> f64;
        pub(crate) safe fn asinf(n: f32) -> f32;
        pub(crate) safe fn atan(x: f64) -> f64;
        pub(crate) safe fn atan2(x: f64, y: f64) -> f64;
        pub(crate) safe fn atan2f(a: f32, b: f32) -> f32;
        pub(crate) safe fn atanf(n: f32) -> f32;
        pub(crate) safe fn cos(x: f64) -> f64;
        pub(crate) safe fn cosf(x: f32) -> f32;
        pub(crate) safe fn cosh(x: f64) -> f64;
        pub(crate) safe fn coshf(n: f32) -> f32;
        pub(crate) safe fn erf(x: f64) -> f64;
        pub(crate) safe fn erfc(x: f64) -> f64;
        pub(crate) safe fn erfcf(x: f32) -> f32;
        pub(crate) safe fn erff(x: f32) -> f32;
        pub(crate) safe fn exp(x: f64) -> f64;
        pub(crate) safe fn exp2(x: f64) -> f64;
        pub(crate) safe fn exp2f(x: f32) -> f32;
        pub(crate) safe fn expf(x: f32) -> f32;
        pub(crate) safe fn expm1(x: f64) -> f64;
        pub(crate) safe fn expm1f(n: f32) -> f32;
        pub(crate) safe fn hypot(x: f64, y: f64) -> f64;
        pub(crate) safe fn hypotf(x: f32, y: f32) -> f32;
        pub(crate) safe fn ldexp(f: f64, n: i32) -> f64;
        pub(crate) safe fn ldexpf(f: f32, n: i32) -> f32;
        pub(crate) safe fn log(x: f64) -> f64;
        pub(crate) safe fn log10(x: f64) -> f64;
        pub(crate) safe fn log10f(x: f32) -> f32;
        pub(crate) safe fn log1p(x: f64) -> f64;
        pub(crate) safe fn log1pf(n: f32) -> f32;
        pub(crate) safe fn log2(x: f64) -> f64;
        pub(crate) safe fn log2f(x: f32) -> f32;
        pub(crate) safe fn logf(x: f32) -> f32;
        pub(crate) safe fn pow(x: f64, y: f64) -> f64;
        pub(crate) safe fn powf(x: f32, y: f32) -> f32;
        pub(crate) safe fn sin(x: f64) -> f64;
        pub(crate) safe fn sinf(x: f32) -> f32;
        pub(crate) safe fn sinh(x: f64) -> f64;
        pub(crate) safe fn sinhf(n: f32) -> f32;
        pub(crate) safe fn tan(x: f64) -> f64;
        pub(crate) safe fn tanf(n: f32) -> f32;
        pub(crate) safe fn tanh(x: f64) -> f64;
        pub(crate) safe fn tanhf(n: f32) -> f32;
        pub(crate) safe fn tgamma(x: f64) -> f64;
        pub(crate) safe fn tgammaf(x: f32) -> f32;
    }
}

/// These symbols exist on some platforms but do not have a compiler-builtins fallback.
pub(crate) mod maybe_available {
    #[allow(dead_code)]
    unsafe extern "C" {
        pub(crate) safe fn acosf128(x: f128) -> f128;
        pub(crate) safe fn asinf128(x: f128) -> f128;
        pub(crate) safe fn atanf128(x: f128) -> f128;
        pub(crate) safe fn cbrtf128(x: f128) -> f128;
        pub(crate) safe fn cosf128(x: f128) -> f128;
        pub(crate) safe fn erff128(x: f128) -> f128;
        pub(crate) safe fn expf128(x: f128) -> f128;
        pub(crate) safe fn exp2f128(x: f128) -> f128;
        pub(crate) safe fn expm1f128(x: f128) -> f128;
        pub(crate) safe fn hypotf128(x: f128, y: f128) -> f128;
        pub(crate) safe fn ldexpf128(f: f128, n: i32) -> f128;
        pub(crate) safe fn log10f128(x: f128) -> f128;
        pub(crate) safe fn log1pf128(x: f128) -> f128;
        pub(crate) safe fn log2f128(x: f128) -> f128;
        pub(crate) safe fn logf128(x: f128) -> f128;
        pub(crate) safe fn powf128(x: f128, y: f128) -> f128;
        pub(crate) safe fn sinf128(x: f128) -> f128;
        pub(crate) safe fn tanf128(x: f128) -> f128;
        pub(crate) safe fn tanhf128(x: f128) -> f128;
        pub(crate) safe fn tgammaf128(x: f128) -> f128;
    }
}
