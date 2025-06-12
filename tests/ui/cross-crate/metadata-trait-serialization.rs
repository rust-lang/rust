//! Test that trait information (like Copy) is correctly serialized in crate metadata

//@ run-pass
//@ aux-build:kinds_in_metadata.rs

/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

extern crate kinds_in_metadata;

use kinds_in_metadata::f;

pub fn main() {
    f::<isize>();
}
