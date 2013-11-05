// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::c_str::CString;
use std::libc::{c_char, c_int};

extern {
    fn sprintf(s: *mut c_char, format: *c_char, ...) -> c_int;
}

unsafe fn check<T>(expected: &str, f: &fn(*mut c_char) -> T) {
    let mut x = [0i8, ..50];
    f(&mut x[0] as *mut c_char);
    let res = CString::new(&x[0], false);
    assert_eq!(expected, res.as_str().unwrap());
}

#[fixed_stack_segment]
pub fn main() {

    unsafe {
        // Call with just the named parameter
        do "Hello World\n".with_c_str |c| {
            check("Hello World\n", |s| sprintf(s, c));
        }

        // Call with variable number of arguments
        do "%d %f %c %s\n".with_c_str |c| {
            do check("42 42.500000 a %d %f %c %s\n\n") |s| {
                sprintf(s, c, 42i, 42.5f64, 'a' as c_int, c);
            }
        }

        // Make a function pointer
        let x: extern "C" unsafe fn(*mut c_char, *c_char, ...) -> c_int = sprintf;

        // A function that takes a function pointer
        unsafe fn call(p: extern "C" unsafe fn(*mut c_char, *c_char, ...) -> c_int) {
            #[fixed_stack_segment];

            // Call with just the named parameter via fn pointer
            do "Hello World\n".with_c_str |c| {
                check("Hello World\n", |s| p(s, c));
            }

            // Call with variable number of arguments
            do "%d %f %c %s\n".with_c_str |c| {
                do check("42 42.500000 a %d %f %c %s\n\n") |s| {
                    p(s, c, 42i, 42.5f64, 'a' as c_int, c);
                }
            }
        }

        // Pass sprintf directly
        call(sprintf);

        // Pass sprintf indirectly
        call(x);
    }

}
