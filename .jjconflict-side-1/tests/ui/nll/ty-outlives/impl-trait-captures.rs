//@ compile-flags:-Zverbose-internals

#![allow(warnings)]

trait Foo<'a> {
}

impl<'a, T> Foo<'a> for T { }

fn foo<'a, T>(x: &T) -> impl Foo<'a> {
    x
    //~^ ERROR captures lifetime that does not appear in bounds
}

fn main() {}
