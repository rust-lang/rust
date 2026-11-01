#![feature(type_alias_impl_trait)]

type Foo = impl Sized;

fn foo<const C: u32>() {}

#[define_opaque(Foo)]
const fn baz() -> Foo {
    42
}

#[define_opaque(Foo)]
fn bar() {
    foo::<{ baz() }>();
    //~^ ERROR: mismatched types
}

fn main() {}
