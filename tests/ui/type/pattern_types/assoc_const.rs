#![feature(pattern_types)]
#![feature(pattern_type_macro)]
#![cfg_attr(const_arg, feature(generic_const_exprs))]
#![expect(incomplete_features)]

//@ revisions: default const_arg

//@[const_arg] check-pass

use std::pat::pattern_type;

trait Foo {
    const START: u32;
    const END: u32;
}

fn foo<T: Foo>(_: pattern_type!(u32 is <T as Foo>::START..=<T as Foo>::END)) {}
//[default]~^ ERROR: constant expression depends on a generic parameter
//[default]~| ERROR: constant expression depends on a generic parameter
fn bar<T: Foo>(_: pattern_type!(u32 is T::START..=T::END)) {}
//[default]~^ ERROR: constant expression depends on a generic parameter
//[default]~| ERROR: constant expression depends on a generic parameter

fn main() {}
