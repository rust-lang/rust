#![feature(rustc_private)]
//@ edition: 2021

// Checks the error messages produced by `#[derive(TryFromU32)]`.

extern crate rustc_macros;

use rustc_macros::TryFromU32;

#[derive(TryFromU32)]
struct MyStruct {} //~ ERROR type is not an enum

#[derive(TryFromU32)]
enum NonTrivial {
    A,
    B(),
    C {},
    D(bool),                //~ ERROR enum variant cannot have fields
    E(bool, bool),          //~ ERROR enum variant cannot have fields
    F { x: bool },          //~ ERROR enum variant cannot have fields
    G { x: bool, y: bool }, //~ ERROR enum variant cannot have fields
}

fn main() {}
