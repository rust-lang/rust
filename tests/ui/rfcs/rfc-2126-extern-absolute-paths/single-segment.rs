//@ aux-build:xcrate.rs
//@ compile-flags:--extern xcrate
//@ edition:2018

use crate; //~ ERROR imports need to be explicitly named
use *; //~ ERROR cannot glob-import all possible crates

fn main() {
    let s = ::xcrate; //~ ERROR cannot find crate `xcrate` in the list of imported crates
                      //~^ NOTE not found in the list of imported crates
                      //~| NOTE a crate named `::xcrate` exists in another namespace
}
