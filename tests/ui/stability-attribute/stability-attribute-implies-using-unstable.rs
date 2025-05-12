//@ aux-build:stability-attribute-implies.rs
#![deny(stable_features)]
#![feature(foo)]
//~^ ERROR the feature `foo` has been partially stabilized since 1.62.0 and is succeeded by the feature `foobar`

// Tests that the use of `implied_by` in the `#[unstable]` attribute results in a diagnostic
// mentioning partial stabilization and that given the implied unstable feature is used (there is a
// `foobar` call), that the compiler suggests changing to that feature and doesn't error about its
// use.

extern crate stability_attribute_implies;
use stability_attribute_implies::{foo, foobar};

fn main() {
    foo();
    foobar(); // no error!
}
