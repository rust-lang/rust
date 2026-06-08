//@ compile-flags: -Zvalidate-mir -Znext-solver

#![feature(min_generic_const_args)]

type const X: usize = const { N };
//~^ ERROR type annotations needed
//~| ERROR type annotations needed

type const N: usize = "this isn't a usize";
//~^ ERROR the constant `"this isn't a usize"` is not of type `usize`

fn main() {}
