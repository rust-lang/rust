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

use std::ffi::{self, CString};
use libc::{c_char, c_int};

// ignore-fast doesn't like extern crate

extern {
    fn sprintf(s: *mut c_char, format: *const c_char, ...) -> c_int;
}

unsafe fn check<T, F>(expected: &str, f: F) where F: FnOnce(*mut c_char) -> T {
    let mut x = [0 as c_char; 50];
    f(&mut x[0] as *mut c_char);
    assert_eq!(expected.as_bytes(), ffi::c_str_to_bytes(&x.as_ptr()));
}

pub fn main() {

    unsafe {
        // Call with just the named parameter
        let c = CString::new(b"Hello World\n").unwrap();
        check("Hello World\n", |s| sprintf(s, c.as_ptr()));

        // Call with variable number of arguments
        let c = CString::new(b"%d %f %c %s\n").unwrap();
        check("42 42.500000 a %d %f %c %s\n\n", |s| {
            sprintf(s, c.as_ptr(), 42, 42.5f64, 'a' as c_int, c.as_ptr());
        });

        // Make a function pointer
        let x: unsafe extern fn(*mut c_char, *const c_char, ...) -> c_int = sprintf;

        // A function that takes a function pointer
        unsafe fn call(p: unsafe extern fn(*mut c_char, *const c_char, ...) -> c_int) {
            // Call with just the named parameter
            let c = CString::new(b"Hello World\n").unwrap();
            check("Hello World\n", |s| sprintf(s, c.as_ptr()));

            // Call with variable number of arguments
            let c = CString::new(b"%d %f %c %s\n").unwrap();
            check("42 42.500000 a %d %f %c %s\n\n", |s| {
                sprintf(s, c.as_ptr(), 42, 42.5f64, 'a' as c_int, c.as_ptr());
            });
        }

        // Pass sprintf directly
        call(sprintf);

        // Pass sprintf indirectly
        call(x);
    }

}
