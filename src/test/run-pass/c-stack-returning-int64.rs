// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast doesn't like extern crate

extern crate libc;

use std::ffi::CString;

mod mlibc {
    use libc::{c_char, c_long, c_longlong};

    extern {
        pub fn atol(x: *const c_char) -> c_long;
        pub fn atoll(x: *const c_char) -> c_longlong;
    }
}

fn atol(s: String) -> int {
    let c = CString::new(s).unwrap();
    unsafe { mlibc::atol(c.as_ptr()) as int }
}

fn atoll(s: String) -> i64 {
    let c = CString::new(s).unwrap();
    unsafe { mlibc::atoll(c.as_ptr()) as i64 }
}

pub fn main() {
    assert_eq!(atol("1024".to_string()) * 10, atol("10240".to_string()));
    assert!((atoll("11111111111111111".to_string()) * 10) ==
             atoll("111111111111111110".to_string()));
}
