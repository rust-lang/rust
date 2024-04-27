// Test sized-ness checking in substitution within fn bodies..

use std::marker;

// Unbounded.
fn f1<X: ?Sized>(x: &X) {
    f2::<X>(x);
    //~^ ERROR the size for values of type
}
fn f2<X>(x: &X) {
}

// Bounded.
trait T {
    fn foo(&self) { }
}
fn f3<X: ?Sized + T>(x: &X) {
    f4::<X>(x);
    //~^ ERROR the size for values of type
}
fn f4<X: T>(x: &X) {
}

fn f5<Y>(x: &Y) {}
fn f6<X: ?Sized>(x: &X) {}

// Test with unsized struct.
struct S<X: ?Sized> {
    x: X,
}

fn f8<X: ?Sized>(x1: &S<X>, x2: &S<X>) {
    f5(x1);
    //~^ ERROR the size for values of type
    f6(x2); // ok
}

// Test some tuples.
fn f9<X: ?Sized>(x1: Box<S<X>>) {
    f5(&(*x1, 34));
    //~^ ERROR the size for values of type
}

fn f10<X: ?Sized>(x1: Box<S<X>>) {
    f5(&(32, *x1));
    //~^ ERROR the size for values of type
    //~| ERROR the size for values of type
}

pub fn main() {}
