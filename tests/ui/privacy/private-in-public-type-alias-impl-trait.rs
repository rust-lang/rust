//@ build-pass (FIXME(62277): could be check-pass?)
#![feature(impl_trait_in_assoc_type)]
#![feature(type_alias_impl_trait)]
#![deny(private_interfaces, private_bounds)]

pub type Pub = impl Default;

#[derive(Default)]
struct Priv;

#[define_opaque(Pub)]
fn check() -> Pub {
    Priv
}

pub trait Trait {
    type Pub: Default;
    fn method() -> Self::Pub;
}

impl Trait for u8 {
    type Pub = impl Default;
    fn method() -> Self::Pub {
        Priv
    }
}

fn main() {}
