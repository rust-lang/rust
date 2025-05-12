//@ build-pass (FIXME(62277): could be check-pass?)

#![feature(type_alias_impl_trait)]

fn main() {}

type Region<'a> = impl std::fmt::Debug;

#[define_opaque(Region)]
fn region<'b>(a: &'b ()) -> Region<'b> {
    a
}
