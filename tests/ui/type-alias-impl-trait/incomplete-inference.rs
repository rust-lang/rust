#![feature(type_alias_impl_trait)]

type Foo = impl Sized;

#[define_opaque(Foo)]
fn bar() -> Foo {
    None
    //~^ ERROR: type annotations needed [E0282]
}

#[define_opaque(Foo)]
fn baz() -> Foo {
    Some(())
}

fn main() {}
