// known-bug: #110395

#![feature(const_trait_impl, effects)]

pub trait A {}
// FIXME ~^ HELP: mark `A` as const

impl const A for () {}
// FIXME ~^ ERROR: const `impl` for trait `A` which is not marked with `#[const_trait]`

fn main() {}
