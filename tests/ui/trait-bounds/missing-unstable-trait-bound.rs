//! Ensure that on stable we don't suggest restricting with an unsafe trait, and we continue
//! mentioning the rest of the obligation chain.

//@ ignore-wasm32
//@ ignore-wasm64
//@ ignore-backends: gcc
//
//@ revisions: stable nightly
//@[stable] act-as-stable
//@[nightly] only-nightly

pub fn baz<T>(t: std::ops::Range<T>) {
    for _ in t {}
    //~^ ERROR: `std::ops::Range<T>` is not an iterator [E0277]
}
fn main() {}
