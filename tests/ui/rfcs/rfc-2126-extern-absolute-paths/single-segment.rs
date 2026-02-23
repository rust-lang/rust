//@ aux-build:xcrate.rs
//@ compile-flags:--extern xcrate
//@ edition:2018

use crate; //~ ERROR imports need to be explicitly named
use *; //~ ERROR cannot glob-import all possible crates

fn main() {
    let s = ::xcrate; //~ ERROR expected value, found crate `xcrate`
                      //~^ NOTE not a value
}
