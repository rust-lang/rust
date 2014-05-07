// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test `type` local variables.


trait T for type {}

fn f1<type X>(x: &X) {
    let _: X; //~ERROR variable `_` has dynamically sized type `X`
    let _: (int, (X, int)); //~ERROR variable `_` has dynamically sized type `(int,(X,int))`
    let y: X; //~ERROR variable `y` has dynamically sized type `X`
    let y: (int, (X, int)); //~ERROR variable `y` has dynamically sized type `(int,(X,int))`
}
fn f2<type X: T>(x: &X) {
    let _: X; //~ERROR variable `_` has dynamically sized type `X`
    let _: (int, (X, int)); //~ERROR variable `_` has dynamically sized type `(int,(X,int))`
    let y: X; //~ERROR variable `y` has dynamically sized type `X`
    let y: (int, (X, int)); //~ERROR variable `y` has dynamically sized type `(int,(X,int))`
}

fn f3<type X>(x1: Box<X>, x2: Box<X>, x3: Box<X>) {
    let y: X = *x1; //~ERROR variable `y` has dynamically sized type `X`
    let y = *x2; //~ERROR variable `y` has dynamically sized type `X`
    let (y, z) = (*x3, 4); //~ERROR variable `y` has dynamically sized type `X`
}
fn f4<type X: T>(x1: Box<X>, x2: Box<X>, x3: Box<X>) {
    let y: X = *x1;         //~ERROR variable `y` has dynamically sized type `X`
    let y = *x2;            //~ERROR variable `y` has dynamically sized type `X`
    let (y, z) = (*x3, 4); //~ERROR variable `y` has dynamically sized type `X`
}

fn g1<type X>(x: X) {} //~ERROR variable `x` has dynamically sized type `X`
fn g2<type X: T>(x: X) {} //~ERROR variable `x` has dynamically sized type `X`

pub fn main() {
}
