//@ check-pass

#![feature(type_alias_impl_trait)]

type Thing = ();

#[define_opaque(Thing)]
fn main() {}
