// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    fn rust_interesting_average(_: u64, ...) -> f64;
}

pub fn main() {
    // Call without variadic arguments
    unsafe {
        assert!(rust_interesting_average(0).is_nan());
    }

    // Call with direct arguments
    unsafe {
        assert_eq!(rust_interesting_average(1, 10i64, 10.0f64) as i64, 20);
    }

    // Call with named arguments, variable number of them
    let (x1, x2, x3, x4) = (10i64, 10.0f64, 20i64, 20.0f64);
    unsafe {
        assert_eq!(rust_interesting_average(2, x1, x2, x3, x4) as i64, 30);
    }

    // A function that takes a function pointer
    unsafe fn call(fp: unsafe extern fn(u64, ...) -> f64) {
        let (x1, x2, x3, x4) = (10i64, 10.0f64, 20i64, 20.0f64);
        assert_eq!(fp(2, x1, x2, x3, x4) as i64, 30);
    }

    unsafe {
        call(rust_interesting_average);

        // Make a function pointer, pass indirectly
        let x: unsafe extern fn(u64, ...) -> f64 = rust_interesting_average;
        call(x);
    }
}
