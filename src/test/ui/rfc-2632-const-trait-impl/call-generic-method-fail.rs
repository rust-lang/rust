#![feature(const_fn_trait_bound)]
#![feature(const_trait_impl)]

pub const fn equals_self<T: PartialEq>(t: &T) -> bool {
    *t == *t
    //~^ ERROR calls in constant functions are limited to constant functions
}

fn main() {}
