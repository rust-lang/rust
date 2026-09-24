//@ compile-flags: --crate-type lib
//@ check-pass

#![feature(generic_const_parameter_types, min_generic_const_args, min_adt_const_params)]
#![allow(incomplete_features)]

const FOO: usize = 5;
const BAR: usize = 5;

pub trait Trait {
    fn method<const A: [usize; FOO]>();
}

pub struct Foo;

impl Trait for Foo {
    fn method<const A: [usize; BAR]>() {}
}
