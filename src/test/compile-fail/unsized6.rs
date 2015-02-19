// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test `?Sized` local variables.

use std::marker;

trait T : marker::MarkerTrait { }

fn f1<X: ?Sized>(x: &X) {
    let _: X; // <-- this is OK, no bindings created, no initializer.
    let _: (isize, (X, isize)); // same
    let y: X; //~ERROR the trait `core::marker::Sized` is not implemented
    let y: (isize, (X, isize)); //~ERROR the trait `core::marker::Sized` is not implemented
}
fn f2<X: ?Sized + T>(x: &X) {
    let y: X; //~ERROR the trait `core::marker::Sized` is not implemented
    let y: (isize, (X, isize)); //~ERROR the trait `core::marker::Sized` is not implemented
}

fn f3<X: ?Sized>(x1: Box<X>, x2: Box<X>, x3: Box<X>) {
    let y: X = *x1; //~ERROR the trait `core::marker::Sized` is not implemented
    let y = *x2; //~ERROR the trait `core::marker::Sized` is not implemented
    let (y, z) = (*x3, 4); //~ERROR the trait `core::marker::Sized` is not implemented
}
fn f4<X: ?Sized + T>(x1: Box<X>, x2: Box<X>, x3: Box<X>) {
    let y: X = *x1;         //~ERROR the trait `core::marker::Sized` is not implemented
    let y = *x2;            //~ERROR the trait `core::marker::Sized` is not implemented
    let (y, z) = (*x3, 4); //~ERROR the trait `core::marker::Sized` is not implemented
}

fn g1<X: ?Sized>(x: X) {} //~ERROR the trait `core::marker::Sized` is not implemented
fn g2<X: ?Sized + T>(x: X) {} //~ERROR the trait `core::marker::Sized` is not implemented

pub fn main() {
}
