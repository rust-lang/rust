#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// FIXME(fmease): I'd prefer to report a cycle error here instead of an overflow one.

struct T;

impl T {
    type This = Self::This; //~ ERROR overflow evaluating associated type `T::This`
}

fn main() {}
