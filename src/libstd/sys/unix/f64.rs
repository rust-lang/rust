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

use core::f64::{NAN, NEG_INFINITY};

pub mod cmath {
    use libc::{c_double, c_int};

    #[link_name = "m"]
    extern {
        pub fn lgamma_r(n: c_double, sign: &mut c_int) -> c_double;
        pub fn hypot(x: c_double, y: c_double) -> c_double;
    }
}

pub fn ln(x: f64) -> f64 {
    log_wrapper(x, |n| { unsafe { ::intrinsics::logf64(n) } })
}

pub fn log2(x: f64) -> f64 {
    log_wrapper(x,
                |n| {
                    #[cfg(target_os = "android")]
                    return ::sys::android::log2f64(n);
                    #[cfg(not(target_os = "android"))]
                    return unsafe { ::intrinsics::log2f64(n) };
                })
}

pub fn log10(x: f64) -> f64 {
    log_wrapper(x, |n| { unsafe { ::intrinsics::log10f64(n) } })
}

// Solaris/Illumos requires a wrapper around log, log2, and log10 functions
// because of their non-standard behavior (e.g. log(-n) returns -Inf instead
// of expected NaN).
fn log_wrapper<F: Fn(f64) -> f64>(x: f64, log_fn: F) -> f64 {
    if !cfg!(target_os = "solaris") {
        log_fn(x)
    } else {
        if x.is_finite() {
            if x > 0.0 {
                log_fn(x)
            } else if x == 0.0 {
                NEG_INFINITY // log(0) = -Inf
            } else {
                NAN // log(-n) = NaN
            }
        } else if x.is_nan() {
            x // log(NaN) = NaN
        } else if x > 0.0 {
            x // log(Inf) = Inf
        } else {
            NAN // log(-Inf) = NaN
        }
    }
}
