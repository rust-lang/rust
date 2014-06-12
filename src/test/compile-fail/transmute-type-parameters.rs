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
    let _: int = transmute(x);  //~ ERROR cannot transmute
}

unsafe fn g<T>(x: (T, int)) {
    let _: int = transmute(x);  //~ ERROR cannot transmute
}

unsafe fn h<T>(x: [T, ..10]) {
    let _: int = transmute(x);  //~ ERROR cannot transmute
}

struct Bad<T> {
    f: T,
}

unsafe fn i<T>(x: Bad<T>) {
    let _: int = transmute(x);  //~ ERROR cannot transmute
}

enum Worse<T> {
    A(T),
    B,
}

unsafe fn j<T>(x: Worse<T>) {
    let _: int = transmute(x);  //~ ERROR cannot transmute
}

unsafe fn k<T>(x: Option<T>) {
    let _: int = transmute(x);  //~ ERROR cannot transmute
}

fn main() {}
