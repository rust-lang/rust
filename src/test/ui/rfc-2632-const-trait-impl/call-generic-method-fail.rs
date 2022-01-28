#![feature(const_fn_trait_bound)]
#![feature(const_trait_impl)]

pub const fn equals_self<T: PartialEq>(t: &T) -> bool {
    *t == *t
    //~^ ERROR can't compare
    //~| ERROR cannot call non-const
}

fn main() {}
