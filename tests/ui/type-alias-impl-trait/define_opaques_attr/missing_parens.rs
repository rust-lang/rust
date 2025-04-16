#![feature(type_alias_impl_trait)]

#[define_opaque]
//~^ ERROR: expected list of type aliases
fn main() {}
