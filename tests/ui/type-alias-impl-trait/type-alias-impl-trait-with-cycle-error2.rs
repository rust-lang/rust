#![feature(type_alias_impl_trait)]

pub trait Bar<T> {
    type Item;
}

type Foo = impl Bar<Foo, Item = Foo>;

fn crash(x: Foo) -> Foo {
    //~^ ERROR: overflow
    x
}

fn main() {}
