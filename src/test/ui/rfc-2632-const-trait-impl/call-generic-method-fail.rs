#![feature(const_fn)]
#![feature(const_trait_impl)]
#![feature(const_trait_bound_opt_out)]
#![allow(incomplete_features)]

pub const fn equals_self<T: ?const PartialEq>(t: &T) -> bool {
    *t == *t
    //~^ ERROR calls in constant functions are limited to constant functions
}

fn main() {}
