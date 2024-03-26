#![feature(type_alias_impl_trait)]
//@ known-bug: #109268

type Foo = impl Fn() -> Foo;

fn crash(x: Foo) -> Foo {
    x
}

fn main() {}
