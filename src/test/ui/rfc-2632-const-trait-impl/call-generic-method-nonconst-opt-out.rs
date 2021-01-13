// check-pass

#![feature(const_fn)]
#![feature(const_trait_impl)]
#![feature(const_trait_bound_opt_out)]
#![allow(incomplete_features)]

struct S;

impl PartialEq for S {
    fn eq(&self, _: &S) -> bool {
        true
    }
}

const fn equals_self<T: ?const PartialEq>(t: &T) -> bool {
    true
}

pub const EQ: bool = equals_self(&S);

// Calling `equals_self` with a type that only has a non-const impl is fine, because we opted out.

fn main() {}
