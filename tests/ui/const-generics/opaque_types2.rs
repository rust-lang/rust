#![feature(type_alias_impl_trait)]

type Foo = impl Sized;

fn foo<const C: u32>() {}

#[defines(Foo)]
const C: Foo = 42;

#[defines(Foo)]
fn bar() {
    foo::<C>();
    //~^ ERROR: mismatched types
}

fn main() {}
