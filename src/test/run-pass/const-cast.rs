// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern fn foo() {}

static x: *u8 = foo;
static y: *libc::c_void = x as *libc::c_void;
static a: &'static int = &10;
static b: *int = a as *int;

pub fn main() {
    assert_eq!(x as *libc::c_void, y);
    assert_eq!(a as *int, b);
}
