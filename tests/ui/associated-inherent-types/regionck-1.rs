#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct U;

impl U {
    // Don't imply any bounds here.

    type NoTyOutliv<'a, T> = &'a T; //~ ERROR the parameter type `T` may not live long enough
    type NoReOutliv<'a, 'b> = &'a &'b (); //~ ERROR reference has a longer lifetime than the data it references
}

fn main() {}
