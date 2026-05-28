#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

struct A;

impl A {
    type const B = 4;
    //~^ ERROR: missing type for `const` item
    //~| ERROR: type annotations needed for the literal
}

fn main() {}
