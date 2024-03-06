// This is a regression test for issue #104400.

//@ revisions: unconstrained constrained
//@[constrained] check-pass

// Test that we can constrain generic const items that appear inside associated consts by
// adding a (makeshift) "evaluatable"-bound to the item.

#![feature(generic_const_items, generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait {
    const LEN: usize;

    #[cfg(unconstrained)]
    const ARRAY: [i32; Self::LEN]; //[unconstrained]~ ERROR unconstrained generic constant

    #[cfg(constrained)]
    const ARRAY: [i32; Self::LEN]
    where
        [(); Self::LEN]:;
}

impl Trait for () {
    const LEN: usize = 2;
    const ARRAY: [i32; Self::LEN] = [360, 720];
}

fn main() {
    let [_, _] = <() as Trait>::ARRAY;
}
