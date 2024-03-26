#![feature(type_alias_impl_trait)]
//@ known-bug: #109268

type Foo<'a> = impl Fn() -> Foo<'a>;

fn crash<'a>(_: &'a (), x: Foo<'a>) -> Foo<'a> {
    x
}

fn main() {}
