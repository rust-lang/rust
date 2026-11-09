//@ revisions: legal illegal
//@[legal] check-pass

#![feature(view_types)]

use std::ops::Deref;
use std::view::view_type;

struct Foo {
    a: (),
    b: (),
    c: (),
}

struct Bar((), (), ());

struct Baz {}
struct Baz_ {
    a: (),
    b: (),
}

impl Deref for Baz {
    type Target = &'static view_type!(Baz_.{ a });

    fn deref(&self) -> &&'static view_type!(Baz_.{ a }) {
        todo!()
    }
}

// Basic access
fn f(foo: &view_type!(Foo.{ a }), bar: &view_type!(Bar.{ 0 })) {
    #[cfg(legal)]
    let _ = foo.a;
    #[cfg(illegal)]
    let _ = foo.b;
    //[illegal]~^ ERROR field `b` cannot be accessed
    //[illegal]~| NOTE this expression has type &view_type!(Foo.{ a }), of which only `a` can be accessed

    #[cfg(legal)]
    let _ = &bar.0;
    #[cfg(illegal)]
    let _ = &bar.1;
    //[illegal]~^ ERROR field `1` cannot be accessed
    //[illegal]~| NOTE this expression has type &view_type!(Bar.{ 0 }), of which only `0` can be accessed
}

// Works with multiple deref
fn g(foo: &&view_type!(Foo.{ a, b }), bar: &&view_type!(Bar.{ 0, 1 })) {
    #[cfg(legal)]
    let _ = foo.a;
    #[cfg(illegal)]
    let _ = foo.c;
    //[illegal]~^ ERROR field `c` cannot be accessed
    //[illegal]~| NOTE this expression has type &&view_type!(Foo.{ a, b }), of which only `a` and `b` can be accessed

    #[cfg(legal)]
    let _ = &bar.0;
    #[cfg(illegal)]
    let _ = &bar.2;
    //[illegal]~^ ERROR field `2` cannot be accessed
    //[illegal]~| NOTE this expression has type &&view_type!(Bar.{ 0, 1 }), of which only `0` and `1` can be accessed
}

// Works with manual deref implementation
fn h(baz: Baz) {
    #[cfg(legal)]
    let _ = baz.a;

    #[cfg(illegal)]
    let _ = baz.b;
    //[illegal]~^ ERROR field `b` cannot be accessed
    //[illegal]~| NOTE this expression has type Baz, of which only `a` can be accessed
}

fn main() {}
