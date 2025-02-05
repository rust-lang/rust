//@ check-pass

#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// Makes sure we can handle `&pin mut place` and `&pin const place` as sugar for
// `unsafe { Pin::new_unchecked(&mut place) }` and `Pin::new(&place)`.

use std::pin::Pin;

struct Foo;

fn foo(_: Pin<&mut Foo>) {
}

fn foo_const(_: Pin<&Foo>) {
}

fn bar() {
    let mut x: Pin<&mut _> = &pin mut Foo;
    foo(x.as_mut());
    foo(x.as_mut());
    foo_const(x);

    let x: Pin<&_> = &pin const Foo;

    foo_const(x);
    foo_const(x);
}

fn main() {}
