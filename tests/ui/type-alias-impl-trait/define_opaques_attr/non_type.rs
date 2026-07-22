#![feature(type_alias_impl_trait)]

fn foo() {}

#[define_opaque(foo)]
//~^ ERROR: cannot find type alias or associated type with opaqaue types `foo` in this scope
fn main() {}
