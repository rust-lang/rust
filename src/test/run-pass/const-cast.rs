// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate libc;

extern fn foo() {}

static x: extern "C" fn() = foo;
static y: *const libc::c_void = x as *const libc::c_void;
static a: &'static int = &10;
static b: *const int = a as *const int;

pub fn main() {
    assert_eq!(x as *const libc::c_void, y);
    assert_eq!(a as *const int, b);
}
