// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_doc)]
#![allow(dead_code)]

//! Bindings for the C math library (for basic mathematic functions)

// Function names are almost identical to C's libmath, a few have been
// renamed, grep for "rename:"

pub mod c_double {
    use libc::{c_double, c_int};

    #[link_name = "m"]
    extern {
        // Alphabetically sorted by link_name

        pub fn acos(n: c_double) -> c_double;
        pub fn asin(n: c_double) -> c_double;
        pub fn atan(n: c_double) -> c_double;
        pub fn atan2(a: c_double, b: c_double) -> c_double;
        pub fn cbrt(n: c_double) -> c_double;
        pub fn cosh(n: c_double) -> c_double;
        pub fn erf(n: c_double) -> c_double;
        pub fn erfc(n: c_double) -> c_double;
        // rename: for consistency with underscore usage elsewhere
        #[link_name="expm1"]
        pub fn exp_m1(n: c_double) -> c_double;
        // rename: for clarity and consistency with add/sub/mul/div
        #[link_name="fdim"]
        pub fn abs_sub(a: c_double, b: c_double) -> c_double;
        #[link_name="fmax"]
        pub fn fmax(a: c_double, b: c_double) -> c_double;
        #[link_name="fmin"]
        pub fn fmin(a: c_double, b: c_double) -> c_double;
        #[link_name="nextafter"]
        pub fn next_after(x: c_double, y: c_double) -> c_double;
        pub fn frexp(n: c_double, value: &mut c_int) -> c_double;
        pub fn hypot(x: c_double, y: c_double) -> c_double;
        pub fn ldexp(x: c_double, n: c_int) -> c_double;
        #[cfg(unix)]
        #[link_name="lgamma_r"]
        pub fn lgamma(n: c_double, sign: &mut c_int) -> c_double;
        #[cfg(windows)]
        #[link_name="__lgamma_r"]
        pub fn lgamma(n: c_double, sign: &mut c_int) -> c_double;
        // renamed: "logb" /often/ is confused for log2 by beginners
        #[link_name="logb"]
        pub fn log_radix(n: c_double) -> c_double;
        // renamed: to be consitent with log as ln
        #[link_name="log1p"]
        pub fn ln_1p(n: c_double) -> c_double;
        #[link_name="ilogb"]
        pub fn ilog_radix(n: c_double) -> c_int;
        pub fn modf(n: c_double, iptr: &mut c_double) -> c_double;
        // rename: for consistency with logradix
        #[link_name="scalbn"]
        pub fn ldexp_radix(n: c_double, i: c_int) -> c_double;
        pub fn sinh(n: c_double) -> c_double;
        pub fn tan(n: c_double) -> c_double;
        pub fn tanh(n: c_double) -> c_double;
        pub fn tgamma(n: c_double) -> c_double;

        // These are commonly only available for doubles

        pub fn j0(n: c_double) -> c_double;
        pub fn j1(n: c_double) -> c_double;
        pub fn jn(i: c_int, n: c_double) -> c_double;

        pub fn y0(n: c_double) -> c_double;
        pub fn y1(n: c_double) -> c_double;
        pub fn yn(i: c_int, n: c_double) -> c_double;
    }
}

pub mod c_float {
    use libc::{c_float, c_int};

    #[link_name = "m"]
    extern {
        // Alphabetically sorted by link_name

        #[link_name="acosf"]
        pub fn acos(n: c_float) -> c_float;
        #[link_name="asinf"]
        pub fn asin(n: c_float) -> c_float;
        #[link_name="atanf"]
        pub fn atan(n: c_float) -> c_float;
        #[link_name="atan2f"]
        pub fn atan2(a: c_float, b: c_float) -> c_float;
        #[link_name="cbrtf"]
        pub fn cbrt(n: c_float) -> c_float;
        #[link_name="coshf"]
        pub fn cosh(n: c_float) -> c_float;
        #[link_name="erff"]
        pub fn erf(n: c_float) -> c_float;
        #[link_name="erfcf"]
        pub fn erfc(n: c_float) -> c_float;
        #[link_name="expm1f"]
        pub fn exp_m1(n: c_float) -> c_float;
        #[link_name="fdimf"]
        pub fn abs_sub(a: c_float, b: c_float) -> c_float;
        #[link_name="frexpf"]
        pub fn frexp(n: c_float, value: &mut c_int) -> c_float;
        #[link_name="fmaxf"]
        pub fn fmax(a: c_float, b: c_float) -> c_float;
        #[link_name="fminf"]
        pub fn fmin(a: c_float, b: c_float) -> c_float;
        #[link_name="nextafterf"]
        pub fn next_after(x: c_float, y: c_float) -> c_float;
        #[link_name="hypotf"]
        pub fn hypot(x: c_float, y: c_float) -> c_float;
        #[link_name="ldexpf"]
        pub fn ldexp(x: c_float, n: c_int) -> c_float;

        #[cfg(unix)]
        #[link_name="lgammaf_r"]
        pub fn lgamma(n: c_float, sign: &mut c_int) -> c_float;

        #[cfg(windows)]
        #[link_name="__lgammaf_r"]
        pub fn lgamma(n: c_float, sign: &mut c_int) -> c_float;

        #[link_name="logbf"]
        pub fn log_radix(n: c_float) -> c_float;
        #[link_name="log1pf"]
        pub fn ln_1p(n: c_float) -> c_float;
        #[link_name="ilogbf"]
        pub fn ilog_radix(n: c_float) -> c_int;
        #[link_name="modff"]
        pub fn modf(n: c_float, iptr: &mut c_float) -> c_float;
        #[link_name="scalbnf"]
        pub fn ldexp_radix(n: c_float, i: c_int) -> c_float;
        #[link_name="sinhf"]
        pub fn sinh(n: c_float) -> c_float;
        #[link_name="tanf"]
        pub fn tan(n: c_float) -> c_float;
        #[link_name="tanhf"]
        pub fn tanh(n: c_float) -> c_float;
        #[link_name="tgammaf"]
        pub fn tgamma(n: c_float) -> c_float;
    }
}
