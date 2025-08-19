#[rustfmt::skip]
#[allow(dead_code)]
#[allow(unused_imports)]
#[allow(clippy::all)]
#[path = "../../../libm/src/math/mod.rs"]
pub(crate) mod libm_math;

macro_rules! libm_intrinsics {
    ($(fn $fun:ident($($iid:ident : $ity:ty),+) -> $oty:ty;)+) => {
        intrinsics! {
            $(
                pub extern "C" fn $fun($($iid: $ity),+) -> $oty {
                    $crate::math::libm_math::$fun($($iid),+)
                }
            )+
        }
    }
}

/// This set of functions is well tested in `libm` and known to provide similar performance to
/// system `libm`, as well as the same or better accuracy.
pub mod full_availability {
    #[cfg(f16_enabled)]
    libm_intrinsics! {
        fn ceilf16(x: f16) -> f16;
        fn copysignf16(x: f16, y: f16) -> f16;
        fn fabsf16(x: f16) -> f16;
        fn fdimf16(x: f16, y: f16) -> f16;
        fn floorf16(x: f16) -> f16;
        fn fmaxf16(x: f16, y: f16) -> f16;
        fn fmaximumf16(x: f16, y: f16) -> f16;
        fn fminf16(x: f16, y: f16) -> f16;
        fn fminimumf16(x: f16, y: f16) -> f16;
        fn fmodf16(x: f16, y: f16) -> f16;
        fn rintf16(x: f16) -> f16;
        fn roundevenf16(x: f16) -> f16;
        fn roundf16(x: f16) -> f16;
        fn sqrtf16(x: f16) -> f16;
        fn truncf16(x: f16) -> f16;
    }

    /* Weak linkage is unreliable on Windows and Apple, so we don't expose symbols that we know
     * the system libc provides in order to avoid conflicts. */

    #[cfg(all(not(windows), not(target_vendor = "apple")))]
    libm_intrinsics! {
        /* f32 */
        fn cbrtf(n: f32) -> f32;
        fn ceilf(x: f32) -> f32;
        fn copysignf(x: f32, y: f32) -> f32;
        fn fabsf(x: f32) -> f32;
        fn fdimf(a: f32, b: f32) -> f32;
        fn floorf(x: f32) -> f32;
        fn fmaf(x: f32, y: f32, z: f32) -> f32;
        fn fmaxf(x: f32, y: f32) -> f32;
        fn fminf(x: f32, y: f32) -> f32;
        fn fmodf(x: f32, y: f32) -> f32;
        fn rintf(x: f32) -> f32;
        fn roundf(x: f32) -> f32;
        fn sqrtf(x: f32) -> f32;
        fn truncf(x: f32) -> f32;

        /* f64 */
        fn cbrt(x: f64) -> f64;
        fn ceil(x: f64) -> f64;
        fn copysign(x: f64, y: f64) -> f64;
        fn fabs(x: f64) -> f64;
        fn fdim(a: f64, b: f64) -> f64;
        fn floor(x: f64) -> f64;
        fn fma(x: f64, y: f64, z: f64) -> f64;
        fn fmax(x: f64, y: f64) -> f64;
        fn fmin(x: f64, y: f64) -> f64;
        fn fmod(x: f64, y: f64) -> f64;
        fn rint(x: f64) -> f64;
        fn round(x: f64) -> f64;
        fn sqrt(x: f64) -> f64;
        fn trunc(x: f64) -> f64;
    }

    // Windows and MacOS do not yet expose roundeven and IEEE 754-2019 `maximum` / `minimum`,
    // however, so we still provide a fallback.
    libm_intrinsics! {
        fn fmaximum(x: f64, y: f64) -> f64;
        fn fmaximumf(x: f32, y: f32) -> f32;
        fn fminimum(x: f64, y: f64) -> f64;
        fn fminimumf(x: f32, y: f32) -> f32;
        fn roundeven(x: f64) -> f64;
        fn roundevenf(x: f32) -> f32;
    }

    #[cfg(f128_enabled)]
    libm_intrinsics! {
        fn ceilf128(x: f128) -> f128;
        fn copysignf128(x: f128, y: f128) -> f128;
        fn fabsf128(x: f128) -> f128;
        fn fdimf128(x: f128, y: f128) -> f128;
        fn floorf128(x: f128) -> f128;
        fn fmaf128(x: f128, y: f128, z: f128) -> f128;
        fn fmaxf128(x: f128, y: f128) -> f128;
        fn fmaximumf128(x: f128, y: f128) -> f128;
        fn fminf128(x: f128, y: f128) -> f128;
        fn fminimumf128(x: f128, y: f128) -> f128;
        fn fmodf128(x: f128, y: f128) -> f128;
        fn rintf128(x: f128) -> f128;
        fn roundevenf128(x: f128) -> f128;
        fn roundf128(x: f128) -> f128;
        fn sqrtf128(x: f128) -> f128;
        fn truncf128(x: f128) -> f128;
    }
}

/// This group of functions has more performance or precision issues than system versions, or
/// are otherwise less well tested. Provide them only on platforms that have problems with the
/// system `libm`.
///
/// As `libm` improves, more functions will be moved from this group to the first group.
///
/// Do not supply for any of the following:
/// - x86 without sse2 due to ABI issues
///   - <https://github.com/rust-lang/rust/issues/114479>
///   - but exclude UEFI since it is a soft-float target
///     - <https://github.com/rust-lang/rust/issues/128533>
/// - All unix targets (linux, macos, freebsd, android, etc)
/// - wasm with known target_os
#[cfg(not(any(
    all(
        target_arch = "x86",
        not(target_feature = "sse2"),
        not(target_os = "uefi"),
    ),
    unix,
    all(target_family = "wasm", not(target_os = "unknown"))
)))]
pub mod partial_availability {
    #[cfg(not(windows))]
    libm_intrinsics! {
        fn acos(x: f64) -> f64;
        fn acosf(n: f32) -> f32;
        fn asin(x: f64) -> f64;
        fn asinf(n: f32) -> f32;
        fn atan(x: f64) -> f64;
        fn atan2(x: f64, y: f64) -> f64;
        fn atan2f(a: f32, b: f32) -> f32;
        fn atanf(n: f32) -> f32;
        fn cos(x: f64) -> f64;
        fn cosf(x: f32) -> f32;
        fn cosh(x: f64) -> f64;
        fn coshf(n: f32) -> f32;
        fn erf(x: f64) -> f64;
        fn erfc(x: f64) -> f64;
        fn erfcf(x: f32) -> f32;
        fn erff(x: f32) -> f32;
        fn exp(x: f64) -> f64;
        fn exp2(x: f64) -> f64;
        fn exp2f(x: f32) -> f32;
        fn expf(x: f32) -> f32;
        fn expm1(x: f64) -> f64;
        fn expm1f(n: f32) -> f32;
        fn hypot(x: f64, y: f64) -> f64;
        fn hypotf(x: f32, y: f32) -> f32;
        fn ldexp(f: f64, n: i32) -> f64;
        fn ldexpf(f: f32, n: i32) -> f32;
        fn log(x: f64) -> f64;
        fn log10(x: f64) -> f64;
        fn log10f(x: f32) -> f32;
        fn log1p(x: f64) -> f64;
        fn log1pf(n: f32) -> f32;
        fn log2(x: f64) -> f64;
        fn log2f(x: f32) -> f32;
        fn logf(x: f32) -> f32;
        fn pow(x: f64, y: f64) -> f64;
        fn powf(x: f32, y: f32) -> f32;
        fn sin(x: f64) -> f64;
        fn sinf(x: f32) -> f32;
        fn sinh(x: f64) -> f64;
        fn sinhf(n: f32) -> f32;
        fn tan(x: f64) -> f64;
        fn tanf(n: f32) -> f32;
        fn tanh(x: f64) -> f64;
        fn tanhf(n: f32) -> f32;
        fn tgamma(x: f64) -> f64;
        fn tgammaf(x: f32) -> f32;
    }

    // allow for windows (and other targets)
    intrinsics! {
        pub extern "C" fn lgamma_r(x: f64, s: &mut i32) -> f64 {
            let r = super::libm_math::lgamma_r(x);
            *s = r.1;
            r.0
        }

        pub extern "C" fn lgammaf_r(x: f32, s: &mut i32) -> f32 {
            let r = super::libm_math::lgammaf_r(x);
            *s = r.1;
            r.0
        }
    }
}
