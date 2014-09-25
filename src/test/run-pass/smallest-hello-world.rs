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
#![feature(intrinsics, lang_items)]

extern crate libc;

extern { fn puts(s: *const u8); }
extern "rust-intrinsic" { fn transmute<T, U>(t: T) -> U; }

#[lang = "stack_exhausted"] extern fn stack_exhausted() {}
#[lang = "eh_personality"] extern fn eh_personality() {}
#[lang = "fail_fmt"] fn fail_fmt() -> ! { loop {} }

#[start]
#[no_split_stack]
fn main(_: int, _: *const *const u8) -> int {
    unsafe {
        let (ptr, _): (*const u8, uint) = transmute("Hello!\0");
        puts(ptr);
    }
    return 0;
}

