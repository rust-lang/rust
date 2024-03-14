#![feature(type_alias_impl_trait)]
//@ known-bug: #109268

pub trait Bar<T> {
    type Item;
}

type Foo = impl Bar<Foo, Item = Foo>;

fn crash(x: Foo) -> Foo {
    x
}

fn main() {}
