#![feature(type_alias_impl_trait)]

fn foo() {}

#[define_opaque(foo)]
//~^ ERROR: expected type alias or associated type with opaqaue types
fn main() {}
