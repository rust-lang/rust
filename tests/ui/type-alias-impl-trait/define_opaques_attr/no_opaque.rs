#![feature(type_alias_impl_trait)]

type Thing = ();

#[define_opaques(Thing)]
//~^ ERROR item does not contain any opaque types
fn main() {}
