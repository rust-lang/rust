// compile-flags: -Z deduplicate-diagnostics=yes

#![warn(unused_imports)]

use std::str::*;
//~^ NOTE `from_utf8` is imported here, but it is not a macro
//~| NOTE `from_utf8_mut` is imported here, but it is not a derive macro
//~| NOTE `from_utf8_unchecked` is imported here, but it is not an attribute

mod hey {
    pub trait Serialize {}
    pub trait Deserialize {}
}

use hey::{Serialize, Deserialize};
//~^ NOTE `Serialize` is imported here, but it is not a derive macro
//~| NOTE `Deserialize` is imported here, but it is not an attribute

#[derive(Serialize)]
//~^ ERROR cannot find derive macro `Serialize`
struct A;

#[derive(from_utf8_mut)]
//~^ ERROR cannot find derive macro `from_utf8_mut`
struct B;

#[derive(println)]
//~^ ERROR cannot find derive macro `println`
//~| NOTE `println` is in scope, but it is a function-like macro
struct C;

#[Deserialize]
//~^ ERROR cannot find attribute `Deserialize`
struct D;

#[from_utf8_unchecked]
//~^ ERROR cannot find attribute `from_utf8_unchecked`
struct E;

#[println]
//~^ ERROR cannot find attribute `println`
//~| NOTE `println` is in scope, but it is a function-like macro
struct F;

fn main() {
    from_utf8!();
    //~^ ERROR cannot find macro `from_utf8`

    Box!();
    //~^ ERROR cannot find macro `Box`
    //~| NOTE `Box` is in scope, but it is not a macro

    Copy!();
    //~^ ERROR cannot find macro `Copy`
    //~| NOTE `Copy` is in scope, but it is a derive macro

    test!();
    //~^ ERROR cannot find macro `test`
    //~| NOTE `test` is in scope, but it is an attribute
}
