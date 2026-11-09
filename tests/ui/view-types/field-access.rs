//@ revisions: legal illegal
//@[legal] check-pass

#![feature(view_types)]

use std::view::view_type;

struct Foo {
    a: (),
    b: (),
    c: (),
}

// Basic access
fn f(foo: &view_type!(Foo.{ a })) {
    #[cfg(legal)]
    let _ = foo.a;
    #[cfg(illegal)]
    let _ = foo.b;
    //[illegal]~^ ERROR field `b` cannot be accessed
    //[illegal]~| NOTE this expression has type &view_type!(Foo.{ a }), of which only `a` can be accessed

    #[cfg(legal)]
    let _ = &foo.a;
    #[cfg(illegal)]
    let _ = &foo.b;
    //[illegal]~^ ERROR field `b` cannot be accessed
    //[illegal]~| NOTE this expression has type &view_type!(Foo.{ a }), of which only `a` can be accessed
}

// Works with multiple deref
fn g(foo: &&view_type!(Foo.{ a, b })) {
    #[cfg(legal)]
    let _ = foo.a;
    #[cfg(illegal)]
    let _ = foo.c;
    //[illegal]~^ ERROR field `c` cannot be accessed
    //[illegal]~| NOTE this expression has type &&view_type!(Foo.{ a, b }), of which only `a` and `b` can be accessed

    #[cfg(legal)]
    let _ = &foo.a;
    #[cfg(illegal)]
    let _ = &foo.c;
    //[illegal]~^ ERROR field `c` cannot be accessed
    //[illegal]~| NOTE this expression has type &&view_type!(Foo.{ a, b }), of which only `a` and `b` can be accessed
}

fn main() {}
