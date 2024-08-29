// Regression test for #82865.

#![feature(decl_macro)]

use x::y::z; //~ ERROR: failed to resolve: you might be missing crate `x`

macro mac () {
    Box::z //~ ERROR: no function or associated item
}

fn main() {
    mac!();
}
