// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test - FIXME(#8538) some kind of problem linking induced by extern "C" fns

// Smallest hello world with no runtime

#![no_std]

// This is an unfortunate thing to have to do on linux :(
#[cfg(target_os = "linux")]
#[doc(hidden)]
pub mod linkhack {
    #[link_args="-lrustrt -lrt"]
    extern {}
}

extern {
    fn puts(s: *u8);
}

extern "rust-intrinsic" {
    fn transmute<T, U>(t: T) -> U;
}

#[start]
pub fn main(_: int, _: **u8, _: *u8) -> int {
    unsafe {
        let (ptr, _): (*u8, uint) = transmute("Hello!");
        puts(ptr);
    }
    return 0;
}

