// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

use intrinsics;

pub mod cmath {
    use libc::{c_float, c_int};

    extern {
        #[cfg_attr(target_env = "msvc", link_name = "__lgammaf_r")]
        pub fn lgammaf_r(n: c_float, sign: &mut c_int) -> c_float;

        #[cfg_attr(target_env = "msvc", link_name = "_hypotf")]
        pub fn hypotf(x: c_float, y: c_float) -> c_float;
    }

    // See the comments in the `floor` function for why MSVC is special
    // here.
    #[cfg(not(target_env = "msvc"))]
    extern {
        pub fn acosf(n: c_float) -> c_float;
        pub fn asinf(n: c_float) -> c_float;
        pub fn atan2f(a: c_float, b: c_float) -> c_float;
        pub fn atanf(n: c_float) -> c_float;
        pub fn coshf(n: c_float) -> c_float;
        pub fn frexpf(n: c_float, value: &mut c_int) -> c_float;
        pub fn ldexpf(x: c_float, n: c_int) -> c_float;
        pub fn sinhf(n: c_float) -> c_float;
        pub fn tanf(n: c_float) -> c_float;
        pub fn tanhf(n: c_float) -> c_float;
    }

    #[cfg(target_env = "msvc")]
    pub use self::shims::*;
    #[cfg(target_env = "msvc")]
    mod shims {
        use libc::{c_float, c_int};

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
        #[allow(deprecated)]
        pub unsafe fn frexpf(x: c_float, value: &mut c_int) -> c_float {
            let (a, b) = f64::frexp(x as f64);
            *value = b as c_int;
            a as c_float
        }

        #[inline]
        #[allow(deprecated)]
        pub unsafe fn ldexpf(x: c_float, n: c_int) -> c_float {
            f64::ldexp(x as f64, n as isize) as c_float
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
}

pub fn floor(x: f32) -> f32 {
    // On MSVC LLVM will lower many math intrinsics to a call to the
    // corresponding function. On MSVC, however, many of these functions
    // aren't actually available as symbols to call, but rather they are all
    // `static inline` functions in header files. This means that from a C
    // perspective it's "compatible", but not so much from an ABI
    // perspective (which we're worried about).
    //
    // The inline header functions always just cast to a f64 and do their
    // operation, so we do that here as well, but only for MSVC targets.
    //
    // Note that there are many MSVC-specific float operations which
    // redirect to this comment, so `floorf` is just one case of a missing
    // function on MSVC, but there are many others elsewhere.
    #[cfg(target_env = "msvc")]
    return (x as f64).floor() as f32;
    #[cfg(not(target_env = "msvc"))]
    return unsafe { intrinsics::floorf32(x) };
}

#[inline]
pub fn ceil(x: f32) -> f32 {
    // see notes above in `floor`
    #[cfg(target_env = "msvc")]
    return (x as f64).ceil() as f32;
    #[cfg(not(target_env = "msvc"))]
    return unsafe { intrinsics::ceilf32(x) };
}

#[inline]
pub fn powf(x: f32, n: f32) -> f32 {
    // see notes above in `floor`
    #[cfg(target_env = "msvc")]
    return (x as f64).powf(n as f64) as f32;
    #[cfg(not(target_env = "msvc"))]
    return unsafe { intrinsics::powf32(x, n) };
}

#[inline]
pub fn exp(x: f32) -> f32 {
    // see notes above in `floor`
    #[cfg(target_env = "msvc")]
    return (x as f64).exp() as f32;
    #[cfg(not(target_env = "msvc"))]
    return unsafe { intrinsics::expf32(x) };
}

#[inline]
pub fn ln(x: f32) -> f32 {
    // see notes above in `floor`
    #[cfg(target_env = "msvc")]
    return (x as f64).ln() as f32;
    #[cfg(not(target_env = "msvc"))]
    return unsafe { intrinsics::logf32(x) };
}

#[inline]
pub fn log2(x: f32) -> f32 {
    unsafe { intrinsics::log2f32(x) }
}

#[inline]
pub fn log10(x: f32) -> f32 {
    // see notes above in `floor`
    #[cfg(target_env = "msvc")]
    return (x as f64).log10() as f32;
    #[cfg(not(target_env = "msvc"))]
    return unsafe { intrinsics::log10f32(x) };
}

#[inline]
pub fn sin(x: f32) -> f32 {
    // see notes in `core::f32::Float::floor`
    #[cfg(target_env = "msvc")]
    return (x as f64).sin() as f32;
    #[cfg(not(target_env = "msvc"))]
    return unsafe { intrinsics::sinf32(x) };
}

#[inline]
pub fn cos(x: f32) -> f32 {
    // see notes in `core::f32::Float::floor`
    #[cfg(target_env = "msvc")]
    return (x as f64).cos() as f32;
    #[cfg(not(target_env = "msvc"))]
    return unsafe { intrinsics::cosf32(x) };
}
