//@ compile-flags: -Z deduplicate-diagnostics=yes

#![warn(unused_imports)]

use std::str::*;
//~^ NOTE `from_utf8` is imported here, but it is a function
//~| NOTE `from_utf8_mut` is imported here, but it is a function
//~| NOTE `from_utf8_unchecked` is imported here, but it is a function

mod hey {
    pub trait Serialize {}
    pub trait Deserialize {}

    pub struct X(i32);
}

use hey::{Serialize, Deserialize, X};
//~^ NOTE `Serialize` is imported here, but it is only a trait, without a derive macro
//~| NOTE `Deserialize` is imported here, but it is a trait
//~| NOTE `X` is imported here, but it is a struct

#[derive(Serialize)]
//~^ ERROR cannot find derive macro `Serialize`
//~| NOTE not found
struct A;

#[derive(from_utf8_mut)]
//~^ ERROR cannot find derive macro `from_utf8_mut`
//~| NOTE not found
struct B;

#[derive(println)]
//~^ ERROR cannot find derive macro `println`
//~| NOTE `println` is in scope, but it is a function-like macro
//~| NOTE not found
struct C;

#[Deserialize]
//~^ ERROR cannot find attribute `Deserialize`
//~| NOTE not found
struct D;

#[from_utf8_unchecked]
//~^ ERROR cannot find attribute `from_utf8_unchecked`
//~| NOTE not found
struct E;

#[println]
//~^ ERROR cannot find attribute `println`
//~| NOTE `println` is in scope, but it is a function-like macro
//~| NOTE not found
struct F;

fn main() {
    from_utf8!();
    //~^ ERROR cannot find macro `from_utf8`
    //~| NOTE not found

    Box!();
    //~^ ERROR cannot find macro `Box`
    //~| NOTE `Box` is in scope, but it is a struct
    //~| NOTE not found

    Copy!();
    //~^ ERROR cannot find macro `Copy`
    //~| NOTE `Copy` is in scope, but it is a derive macro
    //~| NOTE not found

    test!();
    //~^ ERROR cannot find macro `test`
    //~| NOTE `test` is in scope, but it is an attribute
    //~| NOTE not found

    X!();
    //~^ ERROR cannot find macro `X`
    //~| NOTE not found
}
