// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that are conservative around thin/fat pointer mismatches.

#![allow(dead_code)]

use std::mem::transmute;

fn a<T, U: ?Sized>(x: &[T]) -> &U {
    unsafe { transmute(x) } //~ ERROR transmute called with differently sized types
}

fn b<T: ?Sized, U: ?Sized>(x: &T) -> &U {
    unsafe { transmute(x) } //~ ERROR transmute called with differently sized types
}

fn c<T, U>(x: &T) -> &U {
    unsafe { transmute(x) }
}

fn d<T, U>(x: &[T]) -> &[U] {
    unsafe { transmute(x) }
}

fn e<T: ?Sized, U>(x: &T) -> &U {
    unsafe { transmute(x) } //~ ERROR transmute called with differently sized types
}

fn f<T, U: ?Sized>(x: &T) -> &U {
    unsafe { transmute(x) } //~ ERROR transmute called with differently sized types
}

fn main() { }
