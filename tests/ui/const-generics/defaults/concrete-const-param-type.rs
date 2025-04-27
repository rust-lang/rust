#![feature(generic_const_parameter_types, unsized_const_params, adt_const_params)]
//~^ WARN the feature `generic_const_parameter_types` is incomplete
//~| WARN the feature `unsized_const_params` is incomplete
// Make sure that we test the const param type of default const parameters
// if both the type of the default and the type of the parameter are concrete.

use std::marker::ConstParamTy_;

struct Foo<const N: u32, const M: u64 = N>; //~ ERROR the constant `N` is not of type `u64`
struct Bar<T: ConstParamTy_, const N: T, const M: u64 = N>(T); // ok
struct Baz<T: ConstParamTy_, const N: u32, const M: T = N>(T); // ok

fn main() {}
