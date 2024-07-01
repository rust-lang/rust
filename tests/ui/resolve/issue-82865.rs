// Regression test for #82865.

#![feature(decl_macro)]

use x::y::z; //~ ERROR: cannot find item `x`

macro mac () {
    Box::z //~ ERROR: no function or associated item
}

fn main() {
    mac!();
}
