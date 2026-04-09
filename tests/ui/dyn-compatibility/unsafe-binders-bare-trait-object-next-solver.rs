//! Regression test for <https://github.com/rust-lang/rust/issues/151311>
//@ edition: 2024
//@ compile-flags: -Znext-solver=globally

#![feature(unsafe_binders)]

use std::ops::Deref;

trait Foo: Deref<Target = unsafe<'a> &'a dyn Bar> {
    fn method(self: &unsafe<'ops> &'a Bar) {}
    //~^ ERROR expected a type, found a trait
    //~| ERROR use of undeclared lifetime name `'a`
}

trait Bar {}

fn test(x: &dyn Foo) {
    //~^ ERROR the trait `Foo` is not dyn compatible
    x.method();
    //~^ ERROR no method named `method` found for reference `&dyn Foo`
}

fn main() {}
