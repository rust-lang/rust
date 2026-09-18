#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

struct A;

impl A {
    const B = core::direct_const_arg!(4);
    //~^ ERROR: missing type for `const` item
    //~| ERROR: type annotations needed for the literal
}

fn main() {}
