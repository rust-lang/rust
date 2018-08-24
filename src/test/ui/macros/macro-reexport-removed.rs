// aux-build:two_macros.rs

#![feature(macro_reexport)] //~ ERROR feature has been removed

#[macro_reexport(macro_one)] //~ ERROR attribute `macro_reexport` is currently unknown
extern crate two_macros;

fn main() {}
