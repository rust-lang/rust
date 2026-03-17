//! This test ensures compilation failure when trying to pass literals
//! without explicitly stated type as inference variables in generic arguments.
//!
//! See https://github.com/rust-lang/rust/pull/153557

#![allow(incomplete_features)]
#![feature(adt_const_params, min_generic_const_args, generic_const_parameter_types)]

fn main() {
    foo::<_, { 2 }>();
    //~^ ERROR: type annotations needed for the literal
    let _: PC<_, { 42 }> = PC { a: 1, b: 1 };
    //~^ ERROR: type annotations needed for the literal
}

struct PC<T, const N: T> {
//~^ ERROR: `T` can't be used as a const parameter type [E0741]
    a: T,
}

fn foo<const N: usize, const A: [u8; N]>() {}
