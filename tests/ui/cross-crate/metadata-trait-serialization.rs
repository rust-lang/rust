//! Test that trait information (like Copy) is correctly serialized in crate metadata

//@ run-pass
//@ aux-build:kinds_in_metadata.rs

extern crate kinds_in_metadata;

use kinds_in_metadata::f;

pub fn main() {
    f::<isize>();
}
