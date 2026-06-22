//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

pub trait A {}

const impl A for () {}
//~^ ERROR: const `impl` for trait `A` which is not `const`

fn main() {}
