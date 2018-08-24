// aux-build:derive-a.rs

#![allow(warnings)]

#[macro_use]
extern crate derive_a;

use derive_a::derive_a;
//~^ ERROR: unresolved import `derive_a::derive_a`

fn main() {}
