#![feature(type_alias_impl_trait)]

type Foo<'a> = impl Fn() -> Foo<'a>;
//~^ ERROR: unconstrained opaque type

fn crash<'a>(_: &'a (), x: Foo<'a>) -> Foo<'a> {
    x
}

fn main() {}
