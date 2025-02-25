#![feature(type_alias_impl_trait)]

#[define_opaques(Boom)]
//~^ ERROR: cannot find type alias or associated type
fn main() {}
