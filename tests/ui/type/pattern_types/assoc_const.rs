#![feature(pattern_types)]
#![feature(pattern_type_macro)]
#![cfg_attr(const_arg, feature(generic_const_exprs, generic_pattern_types))]
#![expect(incomplete_features)]
// gate-test-generic_pattern_types line to the test file.
//@ revisions: default const_arg

//@[const_arg] check-pass

use std::pat::pattern_type;

trait Foo {
    const START: u32;
    const END: u32;
}

fn foo<T: Foo>(_: pattern_type!(u32 is <T as Foo>::START..=<T as Foo>::END)) {}
//[default]~^ ERROR: generic parameters may not be used in const operations
//[default]~| ERROR: generic parameters may not be used in const operations
fn bar<T: Foo>(_: pattern_type!(u32 is T::START..=T::END)) {}
//[default]~^ ERROR: generic parameters may not be used in const operations
//[default]~| ERROR: generic parameters may not be used in const operations

fn main() {}
