//@ compile-flags: -Zvalidate-mir -Znext-solver

#![feature(min_generic_const_args)]

const X: usize = core::direct_const_arg!(const { N });
//~^ ERROR type annotations needed

const N: usize = core::direct_const_arg!("this isn't a usize");
//~^ ERROR the constant `"this isn't a usize"` is not of type `usize`

fn main() {}
