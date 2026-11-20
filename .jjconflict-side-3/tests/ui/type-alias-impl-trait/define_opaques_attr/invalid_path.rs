#![feature(type_alias_impl_trait)]

#[define_opaque(Boom)]
//~^ ERROR: cannot find type alias or associated type
fn main() {}
