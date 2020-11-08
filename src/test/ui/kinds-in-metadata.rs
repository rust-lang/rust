// run-pass
// aux-build:kinds_in_metadata.rs

// pretty-expanded FIXME #23616

// Tests that metadata serialization works for the `Copy` kind.

extern crate kinds_in_metadata;

use kinds_in_metadata::f;

pub fn main() {
    f::<isize>();
}
