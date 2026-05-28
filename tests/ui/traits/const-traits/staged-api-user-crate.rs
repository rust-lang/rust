//@ aux-build: staged-api.rs
extern crate staged_api;

use staged_api::*;

// Const stability has no impact on usage in non-const contexts.
fn non_const_context() {
    Unstable::func();
}

const fn stable_const_context() {
    Unstable::func();
    //~^ ERROR cannot call conditionally-const associated function `<staged_api::Unstable as staged_api::MyTrait>::func` in constant functions
    //~| ERROR `staged_api::MyTrait` is not yet stable as a const trait
}

fn main() {}
