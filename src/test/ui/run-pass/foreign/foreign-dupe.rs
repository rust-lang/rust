// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:foreign_lib.rs
// ignore-wasm32-bare no libc to test ffi with

// Check that we can still call duplicated extern (imported) functions
// which were declared in another crate. See issues #32740 and #32783.


extern crate foreign_lib;

pub fn main() {
    unsafe {
        let x = foreign_lib::rustrt::rust_get_test_int();
        assert_eq!(x, foreign_lib::rustrt2::rust_get_test_int());
        assert_eq!(x as *const _, foreign_lib::rustrt3::rust_get_test_int());
    }
}
