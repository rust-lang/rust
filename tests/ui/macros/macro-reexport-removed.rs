//@ aux-build:two_macros.rs

#![feature(macro_reexport)] //~ ERROR feature has been removed

#[macro_reexport(macro_one)] //~ ERROR cannot find attribute `macro_reexport` in this scope
extern crate two_macros;

fn main() {}
