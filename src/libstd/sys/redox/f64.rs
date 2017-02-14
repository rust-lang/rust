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

pub mod cmath {
    use libc::{c_double, c_int};

    #[link_name = "m"]
    extern {
        pub fn lgamma_r(n: c_double, sign: &mut c_int) -> c_double;
        pub fn hypot(x: c_double, y: c_double) -> c_double;
    }
}

pub fn ln(x: f64) -> f64 {
    unsafe { ::intrinsics::logf64(x) }
}

pub fn log2(x: f64) -> f64 {
    unsafe { ::intrinsics::log2f64(x) }
}

pub fn log10(x: f64) -> f64 {
    unsafe { ::intrinsics::log10f64(x) }
}
