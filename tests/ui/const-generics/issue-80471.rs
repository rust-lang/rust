#![feature(adt_const_params)]
//~^ WARN the feature `adt_const_params` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

#[derive(PartialEq, Eq)]
enum Nat {
    Z,
    S(Box<Nat>),
}

fn foo<const N: Nat>() {}
//~^ ERROR `Box<Nat>` must be annotated with `#[derive(PartialEq, Eq)]` to be used as the type of a const parameter

fn main() {}
