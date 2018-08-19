// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

fn foo() -> Box<impl fmt::Debug+?Sized> {
    let x : Box<[u8]> = Box::new([0]);
    x
}
fn bar() -> Box<impl fmt::Debug+?Sized> {
    let y: Box<fmt::Debug> = Box::new([0]);
    y
}

fn main() {
    let f = foo();
    let b = bar();

    // this is an `*mut [u8]` in practice
    let f_raw : *mut _ = Box::into_raw(f);
    // this is an `*mut fmt::Debug` in practice
    let mut b_raw = Box::into_raw(b);
    // ... and they should not be mixable
    b_raw = f_raw as *mut _; //~ ERROR is invalid
}
