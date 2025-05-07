//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

pub const fn equals_self<T: PartialEq>(t: &T) -> bool {
    *t == *t
    //~^ ERROR cannot call non-const operator in constant functions
}

fn main() {}
