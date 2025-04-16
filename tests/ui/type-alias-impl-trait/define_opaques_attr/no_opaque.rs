#![feature(type_alias_impl_trait)]

type Thing = ();

#[define_opaque(Thing)]
//~^ ERROR item does not contain any opaque types
fn main() {}
