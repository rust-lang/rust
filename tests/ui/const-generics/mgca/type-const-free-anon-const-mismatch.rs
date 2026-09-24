//@ compile-flags: -Zvalidate-mir -Znext-solver

#![feature(min_generic_const_args)]

use std::gca;

const X: usize = gca!(const { N });
//~^ ERROR type annotations needed

const N: usize = gca!("this isn't a usize");
//~^ ERROR the constant `"this isn't a usize"` is not of type `usize`

fn main() {}
