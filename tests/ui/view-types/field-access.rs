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

    #[cfg(legal)]
    let _ = &bar.0;
    #[cfg(illegal)]
    let _ = &bar.1;
}

// Works with multiple deref
fn g(foo: &&view_type!(Foo.{ a, b }), bar: &&view_type!(Bar.{ 0, 1 })) {
    #[cfg(legal)]
    let _ = foo.a;
    #[cfg(illegal)]
    let _ = foo.c;

    #[cfg(legal)]
    let _ = &bar.0;
    #[cfg(illegal)]
    let _ = &bar.2;
}

// Works with manual deref implementation
fn h(baz: Baz) {
    #[cfg(legal)]
    let _ = baz.a;

    #[cfg(illegal)]
    let _ = baz.b;
}

fn main() {}
