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

struct TestStruct {
    x: *const libc::c_void
}

unsafe impl Sync for TestStruct {}

extern fn foo() {}
const x: extern "C" fn() = foo;
static y: TestStruct = TestStruct { x: x as *const libc::c_void };

pub fn main() {
    assert_eq!(x as *const libc::c_void, y.x);
}
