//@ edition:2015
// Regression test for #82865.

#![feature(decl_macro)]

use x::y::z; //~ ERROR: cannot find module or crate `x`

macro mac () {
    Box::z //~ ERROR: no associated function or constant
}

fn main() {
    mac!();
}
