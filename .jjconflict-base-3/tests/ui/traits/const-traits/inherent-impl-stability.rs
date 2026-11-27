//@ aux-build: staged-api.rs
extern crate staged_api;

use staged_api::*;

// Const stability has no impact on usage in non-const contexts.
fn non_const_context() {
    Unstable::inherent_func();
}

const fn stable_const_context() {
    Unstable::inherent_func();
    //~^ ERROR: `staged_api::Unstable::inherent_func` is not yet stable as a const fn
}

fn main() {}
