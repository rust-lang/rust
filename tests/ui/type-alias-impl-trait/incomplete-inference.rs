#![feature(type_alias_impl_trait)]

type Foo = impl Sized;

fn bar() -> Foo {
    None
    //~^ ERROR: type annotations needed [E0282]
}

fn baz() -> Foo {
    Some(())
}

fn main() {}
