//@ check-pass

#![feature(type_alias_impl_trait)]

type Opaque<'lt> = impl Sized + 'lt;

#[define_opaque(Opaque)]
fn test<'a>(
    arg: impl Iterator<Item = &'a u8>,
) -> impl Iterator<Item = Opaque<'a>> {
    arg
}

fn main() {}
