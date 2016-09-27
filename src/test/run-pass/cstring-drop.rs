// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic
// aux-build:allocator-leaking.rs

// Test that `CString::new("hello").unwrap().as_ptr()` pattern
// leads to failure.

extern crate allocator_leaking;

use std::ffi::CString;
use std::ptr;

fn main() {
    let ptr = CString::new("Hello").unwrap().as_ptr();
    // `ptr` is a dangling pointer and reading it is almost always
    // undefined behavior. But we want to prevent the most diabolical
    // kind of UB (apart from nasal demons): reading a value that was
    // previously written. So we make sure that CString zeros the
    // first byte in the `Drop`.
    // To make the test itself UB-free we use a custom allocator
    // which always leaks memory.
    assert_eq!(unsafe { ptr::read(ptr as *const [u8; 6]) } , *b"\0ello\0");
}
