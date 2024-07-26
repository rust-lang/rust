#![feature(type_alias_impl_trait)]

type Foo = impl Sized;

fn foo<const C: u32>() {}

#[define_opaques(Foo)]
const C: Foo = 42;

#[define_opaques(Foo)]
fn bar() {
    foo::<C>();
    //~^ ERROR: mismatched types
}

fn main() {}
