// Regression test for https://github.com/rust-lang/rust/issues/142913.

// A const parameter introduced by `generic_const_parameter_types`,
// whose type mentions an earlier lifetime
// parameter and which carries a default value, used to ICE in `rustc_type_ir`
// with "region parameter out of range when instantiating args". It is now
// rejected with ordinary errors instead of crashing.

#![feature(generic_const_parameter_types)]

struct Variant;

fn foo<'a, const N: &'a Variant = {}>() {}
//~^ ERROR: defaults for generic parameters are not allowed here
//~| ERROR: anonymous constants with lifetimes in their type are not yet supported
//~| ERROR: `&'a Variant` is forbidden as the type of a const generic parameter

fn main() {}
