#![allow(incomplete_features)]
#![feature(adt_const_params, min_generic_const_args, generic_const_parameter_types)]

fn main() {
    foo::<_, { 2 }>();
    //~^ ERROR: inference variables are not supported in constants
    let _: PC<_, { 42 }> = PC { a: 1, b: 1 };
    //~^ ERROR: inference variables are not supported in constants
}

struct PC<T, const N: T> {
//~^ ERROR: `T` can't be used as a const parameter type [E0741]
    a: T,
}

fn foo<const N: usize, const A: [u8; N]>() {}
