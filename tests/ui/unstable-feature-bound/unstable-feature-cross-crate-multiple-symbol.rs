//@ aux-build:unstable_feature.rs
//@ check-pass
#![feature(feat_bar, feat_moo)]
extern crate unstable_feature;
use unstable_feature::{Foo, Bar};

/// Bar::foo() should still be usable even if we enable multiple feature.

fn main() {
    Bar::foo();
}
