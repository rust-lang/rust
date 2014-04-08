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
    f1::<X>(x);
}
fn f2<X>(x: &X) {
    f1::<X>(x);
    f2::<X>(x);
}

// Bounded.
trait T for type {}
fn f3<type X: T>(x: &X) {
    f3::<X>(x);
}
fn f4<X: T>(x: &X) {
    f3::<X>(x);
    f4::<X>(x);
}

// Self type.
trait T2 for type {
    fn f() -> ~Self;
}
struct S;
impl T2 for S {
    fn f() -> ~S {
        ~S
    }
}
fn f5<type X: T2>(x: &X) {
    let _: ~X = T2::f();
}
fn f6<X: T2>(x: &X) {
    let _: ~X = T2::f();
}

trait T3 for type {
    fn f() -> ~Self;
}
impl T3 for S {
    fn f() -> ~S {
        ~S
    }
}
fn f7<type X: T3>(x: &X) {
    // This is valid, but the unsized bound on X is irrelevant because any type
    // which implements T3 must have statically known size.
    let _: ~X = T3::f();
}

trait T4<X> {
    fn m1(x: &T4<X>);
    fn m2(x: &T5<X>);
}
trait T5<type X> {
    fn m1(x: &T4<X>); // not an error (for now)
    fn m2(x: &T5<X>);
}

trait T6<X: T> {
    fn m1(x: &T4<X>);
    fn m2(x: &T5<X>);
}
trait T7<type X: T> {
    fn m1(x: &T4<X>); // not an error (for now)
    fn m2(x: &T5<X>);
}

pub fn main() {
}
