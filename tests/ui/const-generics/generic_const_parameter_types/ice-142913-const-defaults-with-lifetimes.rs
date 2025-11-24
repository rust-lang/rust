#![allow(incomplete_features)]
#![feature(generic_const_parameter_types)]
struct Variant;

fn foo<'a, const N: &'a Variant = {}>() {}
//~^ ERROR: defaults for generic parameters are not allowed here
//~| ERROR: anonymous constants with lifetimes in their type are not yet supported
//~| ERROR: `&'a Variant` is forbidden as the type of a const generic parameter

fn main() {}
