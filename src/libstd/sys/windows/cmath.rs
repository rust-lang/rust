#![cfg(not(test))]

use libc::{c_float, c_double};

#[link_name = "m"]
extern {
    pub fn acos(n: c_double) -> c_double;
    pub fn asin(n: c_double) -> c_double;
    pub fn atan(n: c_double) -> c_double;
    pub fn atan2(a: c_double, b: c_double) -> c_double;
    pub fn cbrt(n: c_double) -> c_double;
    pub fn cbrtf(n: c_float) -> c_float;
    pub fn cosh(n: c_double) -> c_double;
    pub fn expm1(n: c_double) -> c_double;
    pub fn expm1f(n: c_float) -> c_float;
    pub fn fdim(a: c_double, b: c_double) -> c_double;
    pub fn fdimf(a: c_float, b: c_float) -> c_float;
    #[cfg_attr(target_env = "msvc", link_name = "_hypot")]
    pub fn hypot(x: c_double, y: c_double) -> c_double;
    #[cfg_attr(target_env = "msvc", link_name = "_hypotf")]
    pub fn hypotf(x: c_float, y: c_float) -> c_float;
    pub fn log1p(n: c_double) -> c_double;
    pub fn log1pf(n: c_float) -> c_float;
    pub fn sinh(n: c_double) -> c_double;
    pub fn tan(n: c_double) -> c_double;
    pub fn tanh(n: c_double) -> c_double;
}

pub use self::shims::*;

#[cfg(not(target_env = "msvc"))]
mod shims {
    use libc::c_float;

    extern {
        pub fn acosf(n: c_float) -> c_float;
        pub fn asinf(n: c_float) -> c_float;
        pub fn atan2f(a: c_float, b: c_float) -> c_float;
        pub fn atanf(n: c_float) -> c_float;
        pub fn coshf(n: c_float) -> c_float;
        pub fn sinhf(n: c_float) -> c_float;
        pub fn tanf(n: c_float) -> c_float;
        pub fn tanhf(n: c_float) -> c_float;
    }
}

// On MSVC these functions aren't defined, so we just define shims which promote
// everything fo f64, perform the calculation, and then demote back to f32.
// While not precisely correct should be "correct enough" for now.
#[cfg(target_env = "msvc")]
mod shims {
    use libc::c_float;

    #[inline]
    pub unsafe fn acosf(n: c_float) -> c_float {
        f64::acos(n as f64) as c_float
    }

    #[inline]
    pub unsafe fn asinf(n: c_float) -> c_float {
        f64::asin(n as f64) as c_float
    }

    #[inline]
    pub unsafe fn atan2f(n: c_float, b: c_float) -> c_float {
        f64::atan2(n as f64, b as f64) as c_float
    }

    #[inline]
    pub unsafe fn atanf(n: c_float) -> c_float {
        f64::atan(n as f64) as c_float
    }

    #[inline]
    pub unsafe fn coshf(n: c_float) -> c_float {
        f64::cosh(n as f64) as c_float
    }

    #[inline]
    pub unsafe fn sinhf(n: c_float) -> c_float {
        f64::sinh(n as f64) as c_float
    }

    #[inline]
    pub unsafe fn tanf(n: c_float) -> c_float {
        f64::tan(n as f64) as c_float
    }

    #[inline]
    pub unsafe fn tanhf(n: c_float) -> c_float {
        f64::tanh(n as f64) as c_float
    }
}
