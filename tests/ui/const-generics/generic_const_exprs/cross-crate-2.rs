//@ check-pass
//@ aux-build: cross-crate-2.rs

extern crate cross_crate_2;

use cross_crate_2::Foo;

fn main() {
    Foo::<7>::new();
}
