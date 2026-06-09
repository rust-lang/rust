//@ check-pass

#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// Makes sure we can handle `&pin mut self` and `&pin const self` as sugar for
// `self: Pin<&mut Self>` and `self: Pin<&Self>`.

use std::pin::Pin;

struct Foo;

impl Foo {
    fn baz(&pin mut self) {}

    fn baz_const(&pin const self) {}

    fn baz_lt<'a>(&'a pin mut self) {}

    fn baz_const_lt(&'_ pin const self) {}
}

fn foo(_: &pin mut Foo) {}

fn foo_const(_: &pin const Foo) {}

fn bar(x: &pin mut Foo) {
    // For the calls below to work we need to automatically reborrow,
    // as if the user had written `foo(x.as_mut())`.
    foo(x);
    foo(x);

    Foo::baz(x);
    Foo::baz(x);

    // make sure we can reborrow &mut as &.
    foo_const(x);
    Foo::baz_const(x);

    let x: &pin const _ = Pin::new(&Foo);

    foo_const(x); // make sure reborrowing from & to & works.
    foo_const(x);
}

fn main() {}
