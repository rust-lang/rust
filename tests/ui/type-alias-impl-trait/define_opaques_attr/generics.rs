#![feature(type_alias_impl_trait)]

type Tait<T> = impl Sized;
//~^ ERROR: unconstrained opaque type

#[define_opaque(Tait::<()>)]
//~^ ERROR: expected unsuffixed literal
fn foo() {}

#[define_opaque(Tait<()>)]
//~^ ERROR: expected one of `(`, `,`, `::`, or `=`, found `<`
fn main() {}
