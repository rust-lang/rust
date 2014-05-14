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

// The purpose of this test is to check that we can
// successfully (and safely) invoke external, cdecl
// functions from outside the crate.

extern crate foreign_lib;

pub fn main() {
    unsafe {
        let _foo = foreign_lib::rustrt::rust_get_test_int();
    }
}
