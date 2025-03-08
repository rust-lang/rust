//@ aux-build: cross-crate-2.rs

extern crate cross_crate_2;

use cross_crate_2::Foo;

fn main() {
    // There is an error in the aux crate but there is no way to annotate this
    Foo::<7>::new();
}
