//@ revisions: ed2015 ed2021
//@[ed2015] edition: 2015
//@[ed2021] edition: 2021
pub use _::{a, b};
//~^ ERROR expected identifier, found reserved identifier `_`
pub use std::{a, _};
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR unresolved import `std::a`
pub use std::{b, _, c};
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR unresolved imports `std::b`, `std::c`
pub use std::{_, d};
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR unresolved import `std::d`

fn main() {}
