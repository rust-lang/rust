// compile-pass

#![feature(existential_type)]
#![deny(private_in_public)]

pub existential type Pub: Default;

#[derive(Default)]
struct Priv;

fn check() -> Pub {
    Priv
}

fn main() {}
