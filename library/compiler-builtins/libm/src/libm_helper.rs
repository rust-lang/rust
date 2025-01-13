use core::marker::PhantomData;

use crate::*;

/// Generic helper for libm functions, abstracting over f32 and f64. <br/>
/// # Type Parameter:
/// - `T`: Either `f32` or `f64`
///
/// # Examples
/// ```rust
/// use libm::{self, Libm};
///
/// const PI_F32: f32 = 3.1415927410e+00;
/// const PI_F64: f64 = 3.1415926535897931160e+00;
///
/// assert!(Libm::<f32>::cos(0.0f32) == libm::cosf(0.0));
/// assert!(Libm::<f32>::sin(PI_F32) == libm::sinf(PI_F32));
///
/// assert!(Libm::<f64>::cos(0.0f64) == libm::cos(0.0));
/// assert!(Libm::<f64>::sin(PI_F64) == libm::sin(PI_F64));
/// ```
pub struct Libm<T>(PhantomData<T>);

macro_rules! libm_helper {
    ($t:ident, funcs: $funcs:tt) => {
        impl Libm<$t> {
            #![allow(unused_parens)]

            libm_helper! { $funcs }
        }
    };

    ({$($func:tt;)*}) => {
        $(
            libm_helper! { $func }
        )*
    };

    ((fn $func:ident($($arg:ident: $arg_typ:ty),*) -> ($($ret_typ:ty),*); => $libm_fn:ident)) => {
        #[inline(always)]
        pub fn $func($($arg: $arg_typ),*) -> ($($ret_typ),*) {
            $libm_fn($($arg),*)
        }
    };
}

libm_helper! {
    f32,
    funcs: {
        (fn acos(x: f32) -> (f32);                  => acosf);
        (fn acosh(x: f32) -> (f32);                 => acoshf);
        (fn asin(x: f32) -> (f32);                  => asinf);
        (fn asinh(x: f32) -> (f32);                 => asinhf);
        (fn atan(x: f32) -> (f32);                  => atanf);
        (fn atan2(y: f32, x: f32) -> (f32);         => atan2f);
        (fn atanh(x: f32) -> (f32);                 => atanhf);
        (fn cbrt(x: f32) -> (f32);                  => cbrtf);
        (fn ceil(x: f32) -> (f32);                  => ceilf);
        (fn copysign(x: f32, y: f32) -> (f32);      => copysignf);
        (fn cos(x: f32) -> (f32);                   => cosf);
        (fn cosh(x: f32) -> (f32);                  => coshf);
        (fn erf(x: f32) -> (f32);                   => erff);
        (fn erfc(x: f32) -> (f32);                  => erfcf);
        (fn exp(x: f32) -> (f32);                   => expf);
        (fn exp2(x: f32) -> (f32);                  => exp2f);
        (fn exp10(x: f32) -> (f32);                 => exp10f);
        (fn expm1(x: f32) -> (f32);                 => expm1f);
        (fn fabs(x: f32) -> (f32);                  => fabsf);
        (fn fdim(x: f32, y: f32) -> (f32);          => fdimf);
        (fn floor(x: f32) -> (f32);                 => floorf);
        (fn fma(x: f32, y: f32, z: f32) -> (f32);   => fmaf);
        (fn fmax(x: f32, y: f32) -> (f32);          => fmaxf);
        (fn fmin(x: f32, y: f32) -> (f32);          => fminf);
        (fn fmod(x: f32, y: f32) -> (f32);          => fmodf);
        (fn frexp(x: f32) -> (f32, i32);            => frexpf);
        (fn hypot(x: f32, y: f32) -> (f32);         => hypotf);
        (fn ilogb(x: f32) -> (i32);                 => ilogbf);
        (fn j0(x: f32) -> (f32);                    => j0f);
        (fn j1(x: f32) -> (f32);                    => j1f);
        (fn jn(n: i32, x: f32) -> (f32);            => jnf);
        (fn ldexp(x: f32, n: i32) -> (f32);         => ldexpf);
        (fn lgamma_r(x: f32) -> (f32, i32);         => lgammaf_r);
        (fn lgamma(x: f32) -> (f32);                => lgammaf);
        (fn log(x: f32) -> (f32);                   => logf);
        (fn log1p(x: f32) -> (f32);                 => log1pf);
        (fn log2(x: f32) -> (f32);                  => log2f);
        (fn log10(x: f32) -> (f32);                 => log10f);
        (fn modf(x: f32) -> (f32, f32);             => modff);
        (fn nextafter(x: f32, y: f32) -> (f32);     => nextafterf);
        (fn pow(x: f32, y: f32) -> (f32);           => powf);
        (fn remainder(x: f32, y: f32) -> (f32);     => remainderf);
        (fn remquo(x: f32, y: f32) -> (f32, i32);   => remquof);
        (fn rint(x: f32) -> (f32);                  => rintf);
        (fn round(x: f32) -> (f32);                 => roundf);
        (fn scalbn(x: f32, n: i32) -> (f32);        => scalbnf);
        (fn sin(x: f32) -> (f32);                   => sinf);
        (fn sincos(x: f32) -> (f32, f32);           => sincosf);
        (fn sinh(x: f32) -> (f32);                  => sinhf);
        (fn sqrt(x: f32) -> (f32);                  => sqrtf);
        (fn tan(x: f32) -> (f32);                   => tanf);
        (fn tanh(x: f32) -> (f32);                  => tanhf);
        (fn tgamma(x: f32) -> (f32);                => tgammaf);
        (fn trunc(x: f32) -> (f32);                 => truncf);
        (fn y0(x: f32) -> (f32);                    => y0f);
        (fn y1(x: f32) -> (f32);                    => y1f);
        (fn yn(n: i32, x: f32) -> (f32);            => ynf);
    }
}

libm_helper! {
    f64,
    funcs: {
        (fn acos(x: f64) -> (f64);                  => acos);
        (fn acosh(x: f64) -> (f64);                 => acosh);
        (fn asin(x: f64) -> (f64);                  => asin);
        (fn asinh(x: f64) -> (f64);                 => asinh);
        (fn atan(x: f64) -> (f64);                  => atan);
        (fn atan2(y: f64, x: f64) -> (f64);         => atan2);
        (fn atanh(x: f64) -> (f64);                 => atanh);
        (fn cbrt(x: f64) -> (f64);                  => cbrt);
        (fn ceil(x: f64) -> (f64);                  => ceil);
        (fn copysign(x: f64, y: f64) -> (f64);      => copysign);
        (fn cos(x: f64) -> (f64);                   => cos);
        (fn cosh(x: f64) -> (f64);                  => cosh);
        (fn erf(x: f64) -> (f64);                   => erf);
        (fn erfc(x: f64) -> (f64);                  => erfc);
        (fn exp(x: f64) -> (f64);                   => exp);
        (fn exp2(x: f64) -> (f64);                  => exp2);
        (fn exp10(x: f64) -> (f64);                 => exp10);
        (fn expm1(x: f64) -> (f64);                 => expm1);
        (fn fabs(x: f64) -> (f64);                  => fabs);
        (fn fdim(x: f64, y: f64) -> (f64);          => fdim);
        (fn floor(x: f64) -> (f64);                 => floor);
        (fn fma(x: f64, y: f64, z: f64) -> (f64);   => fma);
        (fn fmax(x: f64, y: f64) -> (f64);          => fmax);
        (fn fmin(x: f64, y: f64) -> (f64);          => fmin);
        (fn fmod(x: f64, y: f64) -> (f64);          => fmod);
        (fn frexp(x: f64) -> (f64, i32);            => frexp);
        (fn hypot(x: f64, y: f64) -> (f64);         => hypot);
        (fn ilogb(x: f64) -> (i32);                 => ilogb);
        (fn j0(x: f64) -> (f64);                    => j0);
        (fn j1(x: f64) -> (f64);                    => j1);
        (fn jn(n: i32, x: f64) -> (f64);            => jn);
        (fn ldexp(x: f64, n: i32) -> (f64);         => ldexp);
        (fn lgamma_r(x: f64) -> (f64, i32);         => lgamma_r);
        (fn lgamma(x: f64) -> (f64);                => lgamma);
        (fn log(x: f64) -> (f64);                   => log);
        (fn log1p(x: f64) -> (f64);                 => log1p);
        (fn log2(x: f64) -> (f64);                  => log2);
        (fn log10(x: f64) -> (f64);                 => log10);
        (fn modf(x: f64) -> (f64, f64);             => modf);
        (fn nextafter(x: f64, y: f64) -> (f64);     => nextafter);
        (fn pow(x: f64, y: f64) -> (f64);           => pow);
        (fn remainder(x: f64, y: f64) -> (f64);     => remainder);
        (fn remquo(x: f64, y: f64) -> (f64, i32);   => remquo);
        (fn rint(x: f64) -> (f64);                  => rint);
        (fn round(x: f64) -> (f64);                 => round);
        (fn scalbn(x: f64, n: i32) -> (f64);        => scalbn);
        (fn sin(x: f64) -> (f64);                   => sin);
        (fn sincos(x: f64) -> (f64, f64);           => sincos);
        (fn sinh(x: f64) -> (f64);                  => sinh);
        (fn sqrt(x: f64) -> (f64);                  => sqrt);
        (fn tan(x: f64) -> (f64);                   => tan);
        (fn tanh(x: f64) -> (f64);                  => tanh);
        (fn tgamma(x: f64) -> (f64);                => tgamma);
        (fn trunc(x: f64) -> (f64);                 => trunc);
        (fn y0(x: f64) -> (f64);                    => y0);
        (fn y1(x: f64) -> (f64);                    => y1);
        (fn yn(n: i32, x: f64) -> (f64);            => yn);
    }
}

#[cfg(f16_enabled)]
libm_helper! {
    f16,
    funcs: {
        (fn copysign(x: f16, y: f16) -> (f16);      => copysignf16);
        (fn fabs(x: f16) -> (f16);                  => fabsf16);
        (fn fdim(x: f16, y: f16) -> (f16);          => fdimf16);
    }
}

#[cfg(f128_enabled)]
libm_helper! {
    f128,
    funcs: {
        (fn copysign(x: f128, y: f128) -> (f128);   => copysignf128);
        (fn fabs(x: f128) -> (f128);                => fabsf128);
        (fn fdim(x: f128, y: f128) -> (f128);       => fdimf128);
    }
}
