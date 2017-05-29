// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that `transmute` cannot be called on type parameters.

use std::mem::transmute;

unsafe fn f<T>(x: T) {
    let _: isize = transmute(x);
//~^ ERROR differently sized types: T (size can vary) to isize
}

unsafe fn g<T>(x: (T, isize)) {
    let _: isize = transmute(x);
//~^ ERROR differently sized types: (T, isize) (size can vary because of T) to isize
}

unsafe fn h<T>(x: [T; 10]) {
    let _: isize = transmute(x);
//~^ ERROR differently sized types: [T; 10] (size can vary because of T) to isize
}

struct Bad<T> {
    f: T,
}

unsafe fn i<T>(x: Bad<T>) {
    let _: isize = transmute(x);
//~^ ERROR differently sized types: Bad<T> (size can vary because of T) to isize
}

enum Worse<T> {
    A(T),
    B,
}

unsafe fn j<T>(x: Worse<T>) {
    let _: isize = transmute(x);
//~^ ERROR differently sized types: Worse<T> (size can vary because of T) to isize
}

unsafe fn k<T>(x: Option<T>) {
    let _: isize = transmute(x);
//~^ ERROR differently sized types: std::option::Option<T> (size can vary because of T) to isize
}

fn main() {}
