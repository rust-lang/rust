//@ compile-flags: -Znext-solver
#![feature(const_trait_impl, const_cmp)]

pub const fn equals_self<T: PartialEq>(t: &T) -> bool {
    *t == *t
    //~^ ERROR the trait bound `T: [const] PartialEq` is not satisfied
}

fn main() {}
