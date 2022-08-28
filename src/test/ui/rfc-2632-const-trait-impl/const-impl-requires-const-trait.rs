#![feature(const_trait_impl)]

pub trait A {}
//~^ NOTE: this trait must be annotated with `#[const_trait]`

impl const A for () {}
//~^ ERROR: const `impl`s must be for traits marked with `#[const_trait]`

fn main() {}
