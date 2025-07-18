//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
trait Trait {
    type Assoc<T: [const] Bound>;
}

impl const Trait for () {
    type Assoc<T: [const] Bound> = T;
}

#[const_trait]
trait Bound {}

fn main() {}
