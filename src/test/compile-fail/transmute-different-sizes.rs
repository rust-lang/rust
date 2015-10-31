// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that `transmute` cannot be called on types of different size.

#![allow(warnings)]

use std::mem::transmute;

unsafe fn f() {
    let _: i8 = transmute(16i16);
    //~^ ERROR transmute called with differently sized types
}

unsafe fn g<T>(x: &T) {
    let _: i8 = transmute(x);
    //~^ ERROR transmute called with differently sized types
}

fn main() {}
