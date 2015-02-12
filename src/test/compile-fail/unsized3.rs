// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test sized-ness checking in substitution within fn bodies..

use std::marker;

// Unbounded.
fn f1<X: ?Sized>(x: &X) {
    f2::<X>(x);
    //~^ ERROR the trait `core::marker::Sized` is not implemented
}
fn f2<X>(x: &X) {
}

// Bounded.
trait T {
    fn foo(&self) { }
}
fn f3<X: ?Sized + T>(x: &X) {
    f4::<X>(x);
    //~^ ERROR the trait `core::marker::Sized` is not implemented
}
fn f4<X: T>(x: &X) {
}

// Test with unsized enum.
enum E<X: ?Sized> {
    V(X),
}

fn f5<Y>(x: &Y) {}
fn f6<X: ?Sized>(x: &X) {}
fn f7<X: ?Sized>(x1: &E<X>, x2: &E<X>) {
    f5(x1);
    //~^ ERROR the trait `core::marker::Sized` is not implemented
    f6(x2); // ok
}


// Test with unsized struct.
struct S<X: ?Sized> {
    x: X,
}

fn f8<X: ?Sized>(x1: &S<X>, x2: &S<X>) {
    f5(x1);
    //~^ ERROR the trait `core::marker::Sized` is not implemented
    f6(x2); // ok
}

// Test some tuples.
fn f9<X: ?Sized>(x1: Box<S<X>>, x2: Box<E<X>>) {
    f5(&(*x1, 34));
    //~^ ERROR the trait `core::marker::Sized` is not implemented
}

fn f10<X: ?Sized>(x1: Box<S<X>>, x2: Box<E<X>>) {
    f5(&(32, *x2));
    //~^ ERROR the trait `core::marker::Sized` is not implemented
}

pub fn main() {
}
