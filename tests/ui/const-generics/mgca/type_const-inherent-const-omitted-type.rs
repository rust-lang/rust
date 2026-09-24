#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

use std::gca;

struct A;

impl A {
    const B = gca!(4);
    //~^ ERROR: missing type for `const` item
    //~| ERROR: type annotations needed for the literal
}

fn main() {}
