#![deny(unused)]

#[macro_use] //~ ERROR unused `#[macro_use]` import
extern crate core;

#[macro_use(
    panic //~ ERROR unused `#[macro_use]` import
)]
extern crate core as core_2;

fn main() {}
