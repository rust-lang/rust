// run-pass

#![allow(unconditional_recursion)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// Test sized-ness checking in substitution.

use std::marker;

// Unbounded.
fn f1<X: ?Sized>(x: &X) {
    f1::<X>(x);
}
fn f2<X>(x: &X) {
    f1::<X>(x);
    f2::<X>(x);
}

// Bounded.
trait T { fn dummy(&self) { } }
fn f3<X: T+?Sized>(x: &X) {
    f3::<X>(x);
}
fn f4<X: T>(x: &X) {
    f3::<X>(x);
    f4::<X>(x);
}

// Self type.
trait T2 {
    fn f() -> Box<Self>;
}
struct S;
impl T2 for S {
    fn f() -> Box<S> {
        Box::new(S)
    }
}
fn f5<X: ?Sized+T2>(x: &X) {
    let _: Box<X> = T2::f();
}
fn f6<X: T2>(x: &X) {
    let _: Box<X> = T2::f();
}

trait T3 {
    fn f() -> Box<Self>;
}
impl T3 for S {
    fn f() -> Box<S> {
        Box::new(S)
    }
}
fn f7<X: ?Sized+T3>(x: &X) {
    // This is valid, but the unsized bound on X is irrelevant because any type
    // which implements T3 must have statically known size.
    let _: Box<X> = T3::f();
}

trait T4<X> {
    fn dummy(&self) { }
    fn m1(&self, x: &dyn T4<X>, y: X);
    fn m2(&self, x: &dyn T5<X>, y: X);
}
trait T5<X: ?Sized> {
    fn dummy(&self) { }
    // not an error (for now)
    fn m1(&self, x: &dyn T4<X>);
    fn m2(&self, x: &dyn T5<X>);
}

trait T6<X: T> {
    fn dummy(&self) { }
    fn m1(&self, x: &dyn T4<X>);
    fn m2(&self, x: &dyn T5<X>);
}
trait T7<X: ?Sized+T> {
    fn dummy(&self) { }
    // not an error (for now)
    fn m1(&self, x: &dyn T4<X>);
    fn m2(&self, x: &dyn T5<X>);
}

// The last field in a struct may be unsized
struct S2<X: ?Sized> {
    f: X,
}
struct S3<X: ?Sized> {
    f1: isize,
    f2: X,
}

pub fn main() {
}
