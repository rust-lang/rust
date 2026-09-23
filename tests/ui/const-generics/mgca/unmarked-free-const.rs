// regression test, used to ICE

#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

use std::gca;

const N: usize = 4;

fn main() {
    let x = [(); gca!(N)];
    //~^ ERROR use of `const` in the type system not marked as direct
}
