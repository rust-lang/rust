// Regression test for #142913
// This used to ICE with "region parameter out of range when instantiating args"
// Now it correctly reports errors without crashing.

#![feature(unsized_const_params, adt_const_params, generic_const_parameter_types)]
//~^ WARN the feature `unsized_const_params` is incomplete
//~| WARN the feature `generic_const_parameter_types` is incomplete

fn foo<'a, const N: &'a Variant = { () as isize }>() {}
//~^ ERROR cannot find type `Variant` in this scope
//~| ERROR defaults for generic parameters are not allowed here
//~| ERROR anonymous constants with lifetimes in their type are not yet supported
//~| ERROR non-primitive cast: `()` as `isize`

fn main() {}
