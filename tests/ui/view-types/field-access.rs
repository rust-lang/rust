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

    #[cfg(legal)]
    let _ = &foo.a;
    #[cfg(illegal)]
    let _ = &foo.b;
}

// Works with multiple deref
fn g(foo: &&view_type!(Foo.{ a, b })) {
    #[cfg(legal)]
    let _ = foo.a;
    #[cfg(illegal)]
    let _ = foo.c;

    #[cfg(legal)]
    let _ = &foo.a;
    #[cfg(illegal)]
    let _ = &foo.c;
}

fn main() {}
