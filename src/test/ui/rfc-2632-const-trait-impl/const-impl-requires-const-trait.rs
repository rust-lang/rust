#![feature(const_trait_impl)]

pub trait A {}
//~^ HELP: mark `A` as const

impl const A for () {}
//~^ ERROR: const `impl` for trait `A` which is not marked with `#[const_trait]`

fn main() {}
