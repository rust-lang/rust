#![feature(type_alias_impl_trait)]

pub trait Bar<T> {
    type Item;
}

type Foo = impl Bar<Foo, Item = Foo>;
//~^ ERROR: unconstrained opaque type

fn crash(x: Foo) -> Foo {
    //~^ ERROR: does not constrain `Foo::{opaque#0}`, but has it in its signature
    x
}

fn main() {}
