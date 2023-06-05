// known-bug: #110395
#![feature(const_trait_impl)]

pub const fn equals_self<T: PartialEq>(t: &T) -> bool {
    *t == *t
    // (remove this) ~^ ERROR can't compare
}

fn main() {}
