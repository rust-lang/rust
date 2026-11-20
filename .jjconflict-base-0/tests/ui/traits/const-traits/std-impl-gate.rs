// This tests feature gates for const impls in the standard library.

//@ revisions: stock gated
//@[gated] run-pass

#![cfg_attr(gated, feature(const_trait_impl, const_default))]

fn non_const_context() -> Vec<usize> {
    Default::default()
}

const fn const_context() -> Vec<usize> {
    Default::default()
    //[stock]~^ ERROR cannot call conditionally-const associated function
    //[stock]~| ERROR `Default` is not yet stable as a const trait
}

fn main() {
    const VAL: Vec<usize> = const_context();

    assert_eq!(VAL, non_const_context());
}
