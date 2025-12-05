//! Test that implicit deref patterns respect the recursion limit
#![feature(deref_patterns)]
#![allow(incomplete_features)]
#![recursion_limit = "8"]

use std::ops::Deref;

struct Cyclic;
impl Deref for Cyclic {
    type Target = Cyclic;
    fn deref(&self) -> &Cyclic {
        &Cyclic
    }
}

fn main() {
    match &Box::new(Cyclic) {
        () => {}
        //~^ ERROR: reached the recursion limit while auto-dereferencing `Cyclic`
        //~| ERROR: the trait bound `Cyclic: DerefPure` is not satisfied
        _ => {}
    }
}
