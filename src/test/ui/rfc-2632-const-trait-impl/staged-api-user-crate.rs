// aux-build: staged-api.rs
extern crate staged_api;

use staged_api::*;

// Const stability has no impact on usage in non-const contexts.
fn non_const_context() {
    Unstable::func();
}

const fn stable_const_context() {
    Unstable::func();
    //~^ ERROR cannot call non-const fn `<staged_api::Unstable as staged_api::MyTrait>::func` in constant functions
}

fn main() {}
