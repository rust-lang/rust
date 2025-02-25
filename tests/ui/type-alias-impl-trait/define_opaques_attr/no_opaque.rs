//@ check-pass

#![feature(type_alias_impl_trait)]

type Thing = ();

#[define_opaques(Thing)]
fn main() {}
