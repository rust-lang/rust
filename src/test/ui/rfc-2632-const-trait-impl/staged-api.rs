// revisions: stable unstable

#![cfg_attr(unstable, feature(unstable))] // The feature from the ./auxiliary/staged-api.rs file.
#![feature(const_trait_impl)]
#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

// aux-build: staged-api.rs
extern crate staged_api;

use staged_api::*;

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stable;

#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(stable, rustc_const_stable(feature = "rust1", since = "1.0.0"))]
impl const MyTrait for Stable {
    //[stable]~^ ERROR trait implementations cannot be const stable yet
    //[unstable]~^^ ERROR implementation has missing const stability attribute
    fn func() {}
}

fn non_const_context() {
    Unstable::func();
    Stable::func();
}

#[unstable(feature = "none", issue = "none")]
const fn const_context() {
    Unstable::func();
    //[stable]~^ ERROR `<staged_api::Unstable as staged_api::MyTrait>::func` is not yet stable as a const fn
    Stable::func();
}

fn main() {}
