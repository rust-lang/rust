//@ check-pass

#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

use std::pin::Pin;

pub struct Foo;

impl Foo {
    fn foo(self: Pin<&mut Self>) {
    }

    fn baz(self: Pin<&Self>) {
    }
}

pub fn bar(x: Pin<&mut Foo>) {
    x.foo();
    x.foo(); // for this to work we need to automatically reborrow,
             // as if the user had written `x.as_mut().foo()`.

    Foo::baz(x);

    x.baz();
}

pub fn baaz(x: Pin<&Foo>) {
    x.baz();
    x.baz();
}

fn main() {}
