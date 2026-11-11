//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

const trait Trait {
    type Assoc<T: [const] Bound>;
}

const impl Trait for () {
    type Assoc<T: [const] Bound> = T;
}

const trait Bound {}

fn main() {}
