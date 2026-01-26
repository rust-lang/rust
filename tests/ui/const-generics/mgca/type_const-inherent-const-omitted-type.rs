#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

struct A;

impl A {
    #[type_const]
    const B = 4;
    //~^ ERROR: missing type for `const` item
}

fn main() {}
