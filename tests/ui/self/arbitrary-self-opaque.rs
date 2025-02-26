#![feature(type_alias_impl_trait)]
struct Foo;

type Bar = impl Sized;

impl Foo {
    fn foo(self: Bar) {}
    //~^ ERROR: invalid `self` parameter type: `Bar`
    //~| ERROR: item does not constrain
}

fn main() {}
