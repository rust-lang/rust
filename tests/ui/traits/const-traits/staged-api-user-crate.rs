//@ aux-build: staged-api.rs
extern crate staged_api;

use staged_api::*;

// Const stability has no impact on usage in non-const contexts.
fn non_const_context() {
    Unstable::func();
}

const fn stable_const_context() {
    Unstable::func();
    //~^ ERROR: cannot call conditionally-const associated function `<staged_api::Unstable as staged_api::MyTrait>::func` in constant functions
    //~| ERROR: `staged_api::MyTrait` is not yet stable as a const trait
}

struct S1;
impl MyTrait for S1 {
    fn func() {}
}

// const impls are gated
struct S2;
impl const MyTrait for S2 {
    //~^ ERROR: `staged_api::MyTrait` is not yet stable as a const trait
    //~| ERROR: const trait impls are experimental
    fn func() {}
}

fn bound1<T: MyTrait>() {}
const fn bound2<T: ~const MyTrait>() {}
//~^ ERROR: `staged_api::MyTrait` is not yet stable as a const trait
//~| ERROR: const trait impls are experimental
fn bound3<T: const MyTrait>() {}
//~^ ERROR: `staged_api::MyTrait` is not yet stable as a const trait
//~| ERROR: const trait impls are experimental

fn main() {}
