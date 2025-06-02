//@ aux-build:unstable_feature.rs

//#![feature(impl_stability)]

extern crate unstable_feature;
use unstable_feature::{Foo, Bar};

fn main() {
    //#![unstable_feature_bound(feat_foo)]
    Bar::foo();
}