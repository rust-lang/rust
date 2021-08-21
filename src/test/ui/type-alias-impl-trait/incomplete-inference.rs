#![feature(type_alias_impl_trait)]

type Foo = impl Sized;

fn bar() -> Foo {
    None
    //~^ ERROR: type annotations needed [E0282]
}

fn baz() -> Foo {
    //~^ ERROR: concrete type differs from previous defining opaque type use
    Some(())
}

fn main() {}
