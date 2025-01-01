//! Bindings to Musl math functions (these are built in `build.rs`).

use std::ffi::{c_char, c_int, c_long};

/// Macro for creating bindings and exposing a safe function (since the implementations have no
/// preconditions). Included functions must have correct signatures, otherwise this will be
/// unsound.
macro_rules! functions {
    ( $(
        $( #[$meta:meta] )*
        $pfx_name:ident: $name:ident( $($arg:ident: $aty:ty),+ ) -> $rty:ty;
    )* ) => {
        extern "C" {
            $( fn $pfx_name( $($arg: $aty),+ ) -> $rty; )*
        }

        $(
            // Expose a safe version
            $( #[$meta] )*
            pub fn $name( $($arg: $aty),+ ) -> $rty {
                // SAFETY: FFI calls with no preconditions
                unsafe { $pfx_name( $($arg),+ ) }
            }
        )*

        #[cfg(test)]
        mod tests {
            use super::*;
            use test_support::CallTest;

            $( functions!(
                @single_test
                $name($($arg: $aty),+) -> $rty
            ); )*
        }
    };

    (@single_test
        $name:ident( $($arg:ident: $aty:ty),+ ) -> $rty:ty
    ) => {
        // Run a simple check to ensure we can link and call the function without crashing.
        #[test]
        // FIXME(#309): LE PPC crashes calling some musl functions
        #[cfg_attr(all(target_arch = "powerpc64", target_endian = "little"), ignore)]
        fn $name() {
            <fn($($aty),+) -> $rty>::check(super::$name);
        }
    };
}

#[cfg(test)]
mod test_support {
    use core::ffi::c_char;

    /// Just verify that we are able to call the function.
    pub trait CallTest {
        fn check(f: Self);
    }

    macro_rules! impl_calltest {
        ($( ($($arg:ty),*) -> $ret:ty; )*) => {
            $(
                impl CallTest for fn($($arg),*) -> $ret {
                    fn check(f: Self) {
                        f($(1 as $arg),*);
                    }
                }
            )*
        };
    }

    impl_calltest! {
        (f32) -> f32;
        (f64) -> f64;
        (f32, f32) -> f32;
        (f64, f64) -> f64;
        (i32, f32) -> f32;
        (i32, f64) -> f64;
        (f32, f32, f32) -> f32;
        (f64, f64, f64) -> f64;
        (f32, i32) -> f32;
        (f32, i64) -> f32;
        (f32) -> i32;
        (f64) -> i32;
        (f64, i32) -> f64;
        (f64, i64) -> f64;
    }

    impl CallTest for fn(f32, &mut f32) -> f32 {
        fn check(f: Self) {
            let mut tmp = 0.0;
            f(0.0, &mut tmp);
        }
    }
    impl CallTest for fn(f64, &mut f64) -> f64 {
        fn check(f: Self) {
            let mut tmp = 0.0;
            f(0.0, &mut tmp);
        }
    }
    impl CallTest for fn(f32, &mut i32) -> f32 {
        fn check(f: Self) {
            let mut tmp = 1;
            f(0.0, &mut tmp);
        }
    }
    impl CallTest for fn(f64, &mut i32) -> f64 {
        fn check(f: Self) {
            let mut tmp = 1;
            f(0.0, &mut tmp);
        }
    }
    impl CallTest for fn(f32, f32, &mut i32) -> f32 {
        fn check(f: Self) {
            let mut tmp = 1;
            f(0.0, 0.0, &mut tmp);
        }
    }
    impl CallTest for fn(f64, f64, &mut i32) -> f64 {
        fn check(f: Self) {
            let mut tmp = 1;
            f(0.0, 0.0, &mut tmp);
        }
    }
    impl CallTest for fn(f32, &mut f32, &mut f32) {
        fn check(f: Self) {
            let mut tmp1 = 1.0;
            let mut tmp2 = 1.0;
            f(0.0, &mut tmp1, &mut tmp2);
        }
    }
    impl CallTest for fn(f64, &mut f64, &mut f64) {
        fn check(f: Self) {
            let mut tmp1 = 1.0;
            let mut tmp2 = 1.0;
            f(0.0, &mut tmp1, &mut tmp2);
        }
    }
    impl CallTest for fn(*const c_char) -> f32 {
        fn check(f: Self) {
            f(c"1".as_ptr());
        }
    }
    impl CallTest for fn(*const c_char) -> f64 {
        fn check(f: Self) {
            f(c"1".as_ptr());
        }
    }
}

functions! {
    musl_acos: acos(a: f64) -> f64;
    musl_acosf: acosf(a: f32) -> f32;
    musl_acosh: acosh(a: f64) -> f64;
    musl_acoshf: acoshf(a: f32) -> f32;
    musl_asin: asin(a: f64) -> f64;
    musl_asinf: asinf(a: f32) -> f32;
    musl_asinh: asinh(a: f64) -> f64;
    musl_asinhf: asinhf(a: f32) -> f32;
    musl_atan2: atan2(a: f64, b: f64) -> f64;
    musl_atan2f: atan2f(a: f32, b: f32) -> f32;
    musl_atan: atan(a: f64) -> f64;
    musl_atanf: atanf(a: f32) -> f32;
    musl_atanh: atanh(a: f64) -> f64;
    musl_atanhf: atanhf(a: f32) -> f32;
    musl_cbrt: cbrt(a: f64) -> f64;
    musl_cbrtf: cbrtf(a: f32) -> f32;
    musl_ceil: ceil(a: f64) -> f64;
    musl_ceilf: ceilf(a: f32) -> f32;
    musl_copysign: copysign(a: f64, b: f64) -> f64;
    musl_copysignf: copysignf(a: f32, b: f32) -> f32;
    musl_cos: cos(a: f64) -> f64;
    musl_cosf: cosf(a: f32) -> f32;
    musl_cosh: cosh(a: f64) -> f64;
    musl_coshf: coshf(a: f32) -> f32;
    musl_drem: drem(a: f64, b: f64) -> f64;
    musl_dremf: dremf(a: f32, b: f32) -> f32;
    musl_erf: erf(a: f64) -> f64;
    musl_erfc: erfc(a: f64) -> f64;
    musl_erfcf: erfcf(a: f32) -> f32;
    musl_erff: erff(a: f32) -> f32;
    musl_exp10: exp10(a: f64) -> f64;
    musl_exp10f: exp10f(a: f32) -> f32;
    musl_exp2: exp2(a: f64) -> f64;
    musl_exp2f: exp2f(a: f32) -> f32;
    musl_exp: exp(a: f64) -> f64;
    musl_expf: expf(a: f32) -> f32;
    musl_expm1: expm1(a: f64) -> f64;
    musl_expm1f: expm1f(a: f32) -> f32;
    musl_fabs: fabs(a: f64) -> f64;
    musl_fabsf: fabsf(a: f32) -> f32;
    musl_fdim: fdim(a: f64, b: f64) -> f64;
    musl_fdimf: fdimf(a: f32, b: f32) -> f32;
    musl_finite: finite(a: f64) -> c_int;
    musl_finitef: finitef(a: f32) -> c_int;
    musl_floor: floor(a: f64) -> f64;
    musl_floorf: floorf(a: f32) -> f32;
    musl_fma: fma(a: f64, b: f64, c: f64) -> f64;
    musl_fmaf: fmaf(a: f32, b: f32, c: f32) -> f32;
    musl_fmax: fmax(a: f64, b: f64) -> f64;
    musl_fmaxf: fmaxf(a: f32, b: f32) -> f32;
    musl_fmin: fmin(a: f64, b: f64) -> f64;
    musl_fminf: fminf(a: f32, b: f32) -> f32;
    musl_fmod: fmod(a: f64, b: f64) -> f64;
    musl_fmodf: fmodf(a: f32, b: f32) -> f32;
    musl_frexp: frexp(a: f64, b: &mut c_int) -> f64;
    musl_frexpf: frexpf(a: f32, b: &mut c_int) -> f32;
    musl_hypot: hypot(a: f64, b: f64) -> f64;
    musl_hypotf: hypotf(a: f32, b: f32) -> f32;
    musl_ilogb: ilogb(a: f64) -> c_int;
    musl_ilogbf: ilogbf(a: f32) -> c_int;
    musl_j0: j0(a: f64) -> f64;
    musl_j0f: j0f(a: f32) -> f32;
    musl_j1: j1(a: f64) -> f64;
    musl_j1f: j1f(a: f32) -> f32;
    musl_jn: jn(a: c_int, b: f64) -> f64;
    musl_jnf: jnf(a: c_int, b: f32) -> f32;
    musl_ldexp: ldexp(a: f64, b: c_int) -> f64;
    musl_ldexpf: ldexpf(a: f32, b: c_int) -> f32;
    musl_lgamma: lgamma(a: f64) -> f64;
    musl_lgamma_r: lgamma_r(a: f64, b: &mut c_int) -> f64;
    musl_lgammaf: lgammaf(a: f32) -> f32;
    musl_lgammaf_r: lgammaf_r(a: f32, b: &mut c_int) -> f32;
    musl_log10: log10(a: f64) -> f64;
    musl_log10f: log10f(a: f32) -> f32;
    musl_log1p: log1p(a: f64) -> f64;
    musl_log1pf: log1pf(a: f32) -> f32;
    musl_log2: log2(a: f64) -> f64;
    musl_log2f: log2f(a: f32) -> f32;
    musl_log: log(a: f64) -> f64;
    musl_logb: logb(a: f64) -> f64;
    musl_logbf: logbf(a: f32) -> f32;
    musl_logf: logf(a: f32) -> f32;
    musl_modf: modf(a: f64, b: &mut f64) -> f64;
    musl_modff: modff(a: f32, b: &mut f32) -> f32;

    // FIXME: these need to be unsafe
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    musl_nan: nan(a: *const c_char) -> f64;
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    musl_nanf: nanf(a: *const c_char) -> f32;

    musl_nearbyint: nearbyint(a: f64) -> f64;
    musl_nearbyintf: nearbyintf(a: f32) -> f32;
    musl_nextafter: nextafter(a: f64, b: f64) -> f64;
    musl_nextafterf: nextafterf(a: f32, b: f32) -> f32;
    musl_pow10: pow10(a: f64) -> f64;
    musl_pow10f: pow10f(a: f32) -> f32;
    musl_pow: pow(a: f64, b: f64) -> f64;
    musl_powf: powf(a: f32, b: f32) -> f32;
    musl_remainder: remainder(a: f64, b: f64) -> f64;
    musl_remainderf: remainderf(a: f32, b: f32) -> f32;
    musl_remquo: remquo(a: f64, b: f64, c: &mut c_int) -> f64;
    musl_remquof: remquof(a: f32, b: f32, c: &mut c_int) -> f32;
    musl_rint: rint(a: f64) -> f64;
    musl_rintf: rintf(a: f32) -> f32;
    musl_round: round(a: f64) -> f64;
    musl_roundf: roundf(a: f32) -> f32;
    musl_scalbln: scalbln(a: f64, b: c_long) -> f64;
    musl_scalblnf: scalblnf(a: f32, b: c_long) -> f32;
    musl_scalbn: scalbn(a: f64, b: c_int) -> f64;
    musl_scalbnf: scalbnf(a: f32, b: c_int) -> f32;
    musl_significand: significand(a: f64) -> f64;
    musl_significandf: significandf(a: f32) -> f32;
    musl_sin: sin(a: f64) -> f64;
    musl_sincos: sincos(a: f64, b: &mut f64, c: &mut f64) -> ();
    musl_sincosf: sincosf(a: f32, b: &mut f32, c: &mut f32) -> ();
    musl_sinf: sinf(a: f32) -> f32;
    musl_sinh: sinh(a: f64) -> f64;
    musl_sinhf: sinhf(a: f32) -> f32;
    musl_sqrt: sqrt(a: f64) -> f64;
    musl_sqrtf: sqrtf(a: f32) -> f32;
    musl_tan: tan(a: f64) -> f64;
    musl_tanf: tanf(a: f32) -> f32;
    musl_tanh: tanh(a: f64) -> f64;
    musl_tanhf: tanhf(a: f32) -> f32;
    musl_tgamma: tgamma(a: f64) -> f64;
    musl_tgammaf: tgammaf(a: f32) -> f32;
    musl_trunc: trunc(a: f64) -> f64;
    musl_truncf: truncf(a: f32) -> f32;
    musl_y0: y0(a: f64) -> f64;
    musl_y0f: y0f(a: f32) -> f32;
    musl_y1: y1(a: f64) -> f64;
    musl_y1f: y1f(a: f32) -> f32;
    musl_yn: yn(a: c_int, b: f64) -> f64;
    musl_ynf: ynf(a: c_int, b: f32) -> f32;
}
