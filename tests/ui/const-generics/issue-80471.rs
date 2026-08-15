#![feature(adt_const_params)]

#[derive(PartialEq, Eq)]
enum Nat {
    Z,
    S(Box<Nat>),
}

fn foo<const N: Nat>() {}
//~^ ERROR `Nat` must implement `ConstParamTy` to be used as the type of a const generic parameter

fn main() {}
