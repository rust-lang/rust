#![feature(effects)]
//~^ WARN: the feature `effects` is incomplete

struct A();

impl const Drop for A {}
//~^ ERROR: const trait impls are experimental
//~| const `impl` for trait `Drop` which is not marked with `#[const_trait]`
//~| not all trait items implemented, missing: `drop`

fn main() {}
