//@ check-pass

#![feature(const_trait_impl, effects)]

pub const fn equals_self<T: PartialEq>(t: &T) -> bool {
    *t == *t
    // FIXME(effects) ~^ ERROR mismatched types
    // FIXME(effects): diagnostic
}

fn main() {}
