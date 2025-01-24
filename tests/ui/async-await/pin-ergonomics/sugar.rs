//@ check-pass

#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// Makes sure we can handle `&pin mut T` and `&pin const T` as sugar for `Pin<&mut T>` and
// `Pin<&T>`.

use std::pin::Pin;

struct Foo;

impl Foo {
    fn baz(self: &pin mut Self) {
    }

    fn baz_const(self: &pin const Self) {
    }

    fn baz_lt<'a>(self: &'a pin mut Self) {
    }

    fn baz_const_lt(self: &'_ pin const Self) {
    }
}

fn foo(_: &pin mut Foo) {
}

fn foo_const(x: &pin const Foo) {
}

fn bar(x: &pin mut Foo) {
    foo(x);
    foo(x); // for this to work we need to automatically reborrow,
            // as if the user had written `foo(x.as_mut())`.

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
