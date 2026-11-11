// regression test, used to ICE

#![feature(gca_min_const_items)]
#![allow(incomplete_features)]

use std::gca;

const N: usize = 4;

fn main() {
    let x = [(); gca!(N)];
    //~^ ERROR use of `const` in the type system not marked as direct
}
