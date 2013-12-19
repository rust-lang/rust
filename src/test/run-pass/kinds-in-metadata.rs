// xfail-fast
// aux-build:kinds_in_metadata.rs

/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

// Tests that metadata serialization works for the `Pod` kind.

extern mod kinds_in_metadata;

use kinds_in_metadata::f;

pub fn main() {
    f::<int>();
}

