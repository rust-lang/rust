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

trait T {}

fn f1<W: ?Sized, X: ?Sized, Y: ?Sized, Z: ?Sized>(x: &X) {
    let _: W; // <-- this is OK, no bindings created, no initializer.
    let _: (isize, (X, isize));
    //~^ ERROR `X` does not have a constant size known at compile-time
    let y: Y;
    //~^ ERROR `Y` does not have a constant size known at compile-time
    let y: (isize, (Z, usize));
    //~^ ERROR `Z` does not have a constant size known at compile-time
}
fn f2<X: ?Sized, Y: ?Sized>(x: &X) {
    let y: X;
    //~^ ERROR `X` does not have a constant size known at compile-time
    let y: (isize, (Y, isize));
    //~^ ERROR `Y` does not have a constant size known at compile-time
}

fn f3<X: ?Sized>(x1: Box<X>, x2: Box<X>, x3: Box<X>) {
    let y: X = *x1;
    //~^ ERROR `X` does not have a constant size known at compile-time
    let y = *x2;
    //~^ ERROR `X` does not have a constant size known at compile-time
    let (y, z) = (*x3, 4);
    //~^ ERROR `X` does not have a constant size known at compile-time
}
fn f4<X: ?Sized + T>(x1: Box<X>, x2: Box<X>, x3: Box<X>) {
    let y: X = *x1;
    //~^ ERROR `X` does not have a constant size known at compile-time
    let y = *x2;
    //~^ ERROR `X` does not have a constant size known at compile-time
    let (y, z) = (*x3, 4);
    //~^ ERROR `X` does not have a constant size known at compile-time
}

fn g1<X: ?Sized>(x: X) {}
//~^ ERROR `X` does not have a constant size known at compile-time
fn g2<X: ?Sized + T>(x: X) {}
//~^ ERROR `X` does not have a constant size known at compile-time

pub fn main() {
}
