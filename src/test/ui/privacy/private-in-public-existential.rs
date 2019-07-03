// build-pass (FIXME(62277): could be check-pass?)

#![feature(existential_type)]
#![deny(private_in_public)]

pub existential type Pub: Default;

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
    existential type Pub: Default;
    fn method() -> Self::Pub { Priv }
}

fn main() {}
