// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #46557

#![feature(nll)]
#![allow(dead_code)]

fn gimme_static_mut() -> &'static mut u32 {
    let ref mut x = 1234543; //~ ERROR borrowed value does not live long enough [E0597]
    x
}

fn main() {}
