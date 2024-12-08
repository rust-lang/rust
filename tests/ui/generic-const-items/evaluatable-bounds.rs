// This is a regression test for issue #104400.

//@ run-rustfix

// Test that we can constrain generic const items that appear inside associated consts by
// adding a (makeshift) "evaluatable"-bound to the item, after applying the suggestion.

#![feature(generic_const_items, generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait {
    const LEN: usize;

    const ARRAY: [i32; Self::LEN]; //~ ERROR unconstrained generic constant

}

impl Trait for () {
    const LEN: usize = 2;
    const ARRAY: [i32; Self::LEN] = [360, 720];
}

fn main() {
    let [_, _] = <() as Trait>::ARRAY;
}
