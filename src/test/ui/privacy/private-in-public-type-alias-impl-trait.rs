// build-pass (FIXME(62277): could be check-pass?)

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
#![deny(private_in_public)]

pub type Pub = impl Default;

#[derive(Default)]
struct Priv;

fn check() -> Pub {
    Priv
}

pub trait Trait {
    type Pub: Default;
    fn method() -> Self::Pub;
}

impl Trait for u8 {
    type Pub = impl Default;
    fn method() -> Self::Pub { Priv }
}

fn main() {}
