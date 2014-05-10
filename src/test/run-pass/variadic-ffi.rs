// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate libc;
use std::c_str::CString;
use libc::{c_char, c_int};

// ignore-fast doesn't like extern crate

extern {
    fn sprintf(s: *mut c_char, format: *c_char, ...) -> c_int;
}

unsafe fn check<T>(expected: &str, f: |*mut c_char| -> T) {
    let mut x = [0i8, ..50];
    f(&mut x[0] as *mut c_char);
    let res = CString::new(&x[0], false);
    assert_eq!(expected, res.as_str().unwrap());
}

pub fn main() {

    unsafe {
        // Call with just the named parameter
        "Hello World\n".with_c_str(|c| {
            check("Hello World\n", |s| sprintf(s, c));
        });

        // Call with variable number of arguments
        "%d %f %c %s\n".with_c_str(|c| {
            check("42 42.500000 a %d %f %c %s\n\n", |s| {
                sprintf(s, c, 42i, 42.5f64, 'a' as c_int, c);
            })
        });

        // Make a function pointer
        let x: unsafe extern "C" fn(*mut c_char, *c_char, ...) -> c_int = sprintf;

        // A function that takes a function pointer
        unsafe fn call(p: unsafe extern "C" fn(*mut c_char, *c_char, ...) -> c_int) {
            // Call with just the named parameter via fn pointer
            "Hello World\n".with_c_str(|c| {
                check("Hello World\n", |s| p(s, c));
            });

            // Call with variable number of arguments
            "%d %f %c %s\n".with_c_str(|c| {
                check("42 42.500000 a %d %f %c %s\n\n", |s| {
                    p(s, c, 42i, 42.5f64, 'a' as c_int, c);
                })
            });
        }

        // Pass sprintf directly
        call(sprintf);

        // Pass sprintf indirectly
        call(x);
    }

}
