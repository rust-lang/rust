//@ compile-flags: --crate-type lib
//@ check-pass

#![feature(min_adt_const_params)]
#![feature(generic_const_parameter_types)]
#![allow(incomplete_features)]

pub trait Trait {
    fn method<const A: usize, const B: [usize; A]>();
}

pub struct Foo;

impl Trait for Foo {
    fn method<const A: usize, const B: [usize; A]>() {}
}
