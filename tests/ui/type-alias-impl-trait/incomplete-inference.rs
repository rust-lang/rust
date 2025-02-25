#![feature(type_alias_impl_trait)]

type Foo = impl Sized;

#[define_opaques(Foo)]
fn bar() -> Foo {
    None
    //~^ ERROR: type annotations needed [E0282]
}

#[define_opaques(Foo)]
fn baz() -> Foo {
    Some(())
}

fn main() {}
