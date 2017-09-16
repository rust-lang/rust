// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Static recursion check shouldn't fail when given a foreign item (#18279)

// aux-build:check_static_recursion_foreign_helper.rs

// pretty-expanded FIXME #23616

#![feature(custom_attribute, libc)]

extern crate check_static_recursion_foreign_helper;
extern crate libc;

use libc::c_int;

#[link_name = "check_static_recursion_foreign_helper"]
extern "C" {
    #[allow(dead_code)]
    static test_static: c_int;
}

static B: &'static c_int = unsafe { &test_static };

pub fn main() {}
