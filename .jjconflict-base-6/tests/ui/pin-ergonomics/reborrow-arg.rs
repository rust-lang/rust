//@ check-pass

#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

use std::pin::Pin;

struct Foo;

impl Foo {
    fn baz(self: Pin<&mut Self>) {
    }
}

fn foo(_: Pin<&mut Foo>) {
}

fn foo_const(_: Pin<&Foo>) {
}

fn bar(x: Pin<&mut Foo>) {
    foo(x);
    foo(x); // for this to work we need to automatically reborrow,
            // as if the user had written `foo(x.as_mut())`.

    Foo::baz(x);
    Foo::baz(x);

    foo_const(x); // make sure we can reborrow &mut as &.

    let x: Pin<&Foo> = Pin::new(&Foo);

    foo_const(x); // make sure reborrowing from & to & works.
}

fn main() {}
