// compile-flags:-Zborrowck=mir -Zverbose

#![allow(warnings)]

trait Foo<'a> {
}

impl<'a, T> Foo<'a> for T { }

fn foo<'a, T>(x: &T) -> impl Foo<'a> {
//~^ ERROR captures lifetime that does not appear in bounds
    x
}

fn main() {}
