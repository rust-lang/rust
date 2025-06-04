// Regression test for #82865.

#![feature(decl_macro)]

use x::y::z; //~ ERROR cannot find item `x`
//~^ NOTE use of unresolved module or unlinked crate `x`

macro mac () {
    Box::z //~ ERROR: no function or associated item
    //~^ NOTE function or associated item not found in `Box<_, _>`
}

fn main() {
    mac!(); //~ NOTE in this expansion of mac!
}
