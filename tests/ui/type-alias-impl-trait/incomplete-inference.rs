#![feature(type_alias_impl_trait)]

type Foo = impl Sized;

#[defines(Foo)]
fn bar() -> Foo {
    None
    //~^ ERROR: type annotations needed [E0282]
}

#[defines(Foo)]
fn baz() -> Foo {
    Some(())
}

fn main() {}
