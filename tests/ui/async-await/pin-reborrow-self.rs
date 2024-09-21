//@ check-pass
//@ignore-test

// Currently ignored due to self reborrowing not being implemented for Pin

#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

use std::pin::Pin;

struct Foo;

impl Foo {
    fn foo(self: Pin<&mut Self>) {
    }
}

fn bar(x: Pin<&mut Foo>) {
    x.foo();
    x.foo(); // for this to work we need to automatically reborrow,
             // as if the user had written `x.as_mut().foo()`.
}

fn main() {}
