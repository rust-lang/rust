#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

struct A;

impl A {
    #[type_const]
    const B = 4;
    //~^ ERROR: omitting type on const item declaration is experimental [E0658]
    //~| ERROR: mismatched types [E0308]
}

fn main() {}
