//@ check-pass
//@ compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(const_trait_impl, effects)]

pub const fn equals_self<T: PartialEq>(t: &T) -> bool {
    *t == *t
    // FIXME(effects) ~^ ERROR mismatched types
    // FIXME(effects): diagnostic
}

fn main() {}
