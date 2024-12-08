//@ aux-build:stability-attribute-implies.rs
#![deny(stable_features)]
#![feature(foo)]
//~^ ERROR the feature `foo` has been partially stabilized since 1.62.0 and is succeeded by the feature `foobar`

// Tests that the use of `implied_by` in the `#[unstable]` attribute results in a diagnostic
// mentioning partial stabilization, and that given the implied unstable feature is unused (there
// is no `foobar` call), that the compiler suggests removing the flag.

extern crate stability_attribute_implies;
use stability_attribute_implies::foo;

fn main() {
    foo();
}
