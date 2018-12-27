// Regression test for #29859, initial version. This example allowed
// arbitrary trait bounds to be synthesized.

// Trait that you want all types to implement.
use std::marker::{Sync as Trait};

pub trait Magic {
    type X: Trait;
}
impl<T: Magic> Magic for T {
    type X = Self;
}

fn check<T: Trait>() {}

fn wizard<T: Magic>() { check::<<T as Magic>::X>(); }

fn main() {
    wizard::<*mut ()>(); //~ ERROR E0275
    // check::<*mut ()>();
}
