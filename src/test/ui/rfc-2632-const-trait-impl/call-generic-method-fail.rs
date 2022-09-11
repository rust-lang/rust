#![feature(const_trait_impl)]

pub const fn equals_self<T: PartialEq>(t: &T) -> bool {
    *t == *t
    //~^ ERROR can't compare
}

fn main() {}
