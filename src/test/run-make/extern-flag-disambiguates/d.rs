// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(before)] extern crate a;
extern crate b;
extern crate c;
#[cfg(after)] extern crate a;

fn t(a: &'static uint) -> uint { a as *const _ as uint }

fn main() {
    assert!(t(a::token()) == t(b::a_token()));
    assert!(t(a::token()) != t(c::a_token()));
}
