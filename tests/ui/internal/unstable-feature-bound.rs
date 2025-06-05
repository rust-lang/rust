//@ aux-build:unstable_feature.rs
#![feature(feat_bar)]
extern crate unstable_feature;
use unstable_feature::{Foo, Bar, Moo};

/// Since only `feat_bar` is enabled, Bar::foo() should be usable and
/// Moo::foo() should not be usable.

fn main() {
    Bar::foo();
    Moo::foo();
    //~^ ERROR: type annotations needed: cannot satisfy `unstable feature: `feat_moo``
}
