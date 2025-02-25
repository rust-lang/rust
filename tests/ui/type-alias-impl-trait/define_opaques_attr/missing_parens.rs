#![feature(type_alias_impl_trait)]

#[define_opaques]
//~^ ERROR: expected list of type aliases
fn main() {}
