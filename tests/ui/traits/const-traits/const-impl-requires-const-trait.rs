//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

pub trait A {}

impl const A for () {}
//~^ ERROR: const `impl` for trait `A` which is not marked with `#[const_trait]`

fn main() {}
