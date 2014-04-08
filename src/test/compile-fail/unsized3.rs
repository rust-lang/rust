// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test sized-ness checking in substitution.

// Unbounded.
fn f1<type X>(x: &X) {
    f2::<X>(x); //~ ERROR instantiating a type parameter with an incompatible type `X`, which does n
}
fn f2<X>(x: &X) {
}

// Bounded.
trait T for type {}
fn f3<type X: T>(x: &X) {
    f4::<X>(x); //~ ERROR instantiating a type parameter with an incompatible type `X`, which does n
}
fn f4<X: T>(x: &X) {
}

// Test with unsized enum.
enum E<type X> {
    V(X),
}

fn f5<Y>(x: &Y) {}
fn f6<type X>(x: &X) {}
fn f7<type X>(x1: &E<X>, x2: &E<X>) {
    f5(x1); //~ERROR instantiating a type parameter with an incompatible type `E<X>`, which does not
    f6(x2); // ok
}


// Test with unsized struct.
struct S<type X> {
    x: X,
}

fn f8<type X>(x1: &S<X>, x2: &S<X>) {
    f5(x1); //~ERROR instantiating a type parameter with an incompatible type `S<X>`, which does not
    f6(x2); // ok
}

// Test some tuples.
fn f9<type X>(x1: ~S<X>, x2: ~E<X>) {
    f5(&(*x1, 34)); //~ERROR instantiating a type parameter with an incompatible type `(S<X>,int)`,
    f5(&(32, *x2)); //~ERROR instantiating a type parameter with an incompatible type `(int,E<X>)`,
}

// I would like these to fail eventually.
/*
// impl - bounded
trait T1<Z: T> {
}
struct S3<type Y>;
impl<type X: T> T1<X> for S3<X> { //ERROR instantiating a type parameter with an incompatible type
}

// impl - unbounded
trait T2<Z> {
}
impl<type X> T2<X> for S3<X> { //ERROR instantiating a type parameter with an incompatible type `X`

// impl - struct
trait T3<type Z> {
}
struct S4<Y>;
impl<type X> T3<X> for S4<X> { //ERROR instantiating a type parameter with an incompatible type `X`
}
*/

pub fn main() {
}
