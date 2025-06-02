//@ aux-build:unstable_feature.rs

#![feature(impl_stability)]
#![feature(trivial_bounds)]

extern crate unstable_feature;
use unstable_feature::{Foo, Bar};

#[unstable_feature_bound(feat_foo)]
fn main() { //~ ERROR: `main` function is not allowed to have a `where` clause
    Bar::foo();
}