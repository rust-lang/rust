// revisions: stock staged
#![cfg_attr(staged, feature(staged))]

#![feature(const_trait_impl)]
#![allow(incomplete_features)]

#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

// aux-build: staged-api.rs
extern crate staged_api;

use staged_api::*;

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stable;

#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(staged, rustc_const_stable(feature = "rust1", since = "1.0.0"))]
// ^ should trigger error with or without the attribute
impl const MyTrait for Stable {
    fn func() { //~ ERROR trait methods cannot be stable const fn

    }
}

fn non_const_context() {
    Unstable::func();
    Stable::func();
}

#[unstable(feature = "none", issue = "none")]
const fn const_context() {
    Unstable::func();
    //[stock]~^ ERROR `<staged_api::Unstable as staged_api::MyTrait>::func` is not yet stable as a const fn
    Stable::func();
}

fn main() {}
