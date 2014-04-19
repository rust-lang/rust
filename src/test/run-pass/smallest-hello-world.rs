// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android

// Smallest "hello world" with a libc runtime

#![no_std]

extern crate libc;

extern { fn puts(s: *u8); }
extern "rust-intrinsic" { fn transmute<T, U>(t: T) -> U; }

#[no_mangle]
pub extern fn rust_stack_exhausted() {}

#[start]
#[no_split_stack]
fn main(_: int, _: **u8) -> int {
    unsafe {
        let (ptr, _): (*u8, uint) = transmute("Hello!\0");
        puts(ptr);
    }
    return 0;
}

