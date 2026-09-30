#![expect(incomplete_features)]
#![feature(gca_min_const_items)]

use std::gca;

struct A;

impl A {
    const B = gca!(4);
    //~^ ERROR: missing type for `const` item
    //~| ERROR: type annotations needed for the literal
}

fn main() {}
