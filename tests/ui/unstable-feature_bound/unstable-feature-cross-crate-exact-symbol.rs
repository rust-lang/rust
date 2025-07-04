//@ aux-build:unstable_feature.rs
//@ revisions: pass fail
//@[pass] check-pass

#![cfg_attr(pass, feature(feat_bar, feat_moo))]
#![cfg_attr(fail, feature(feat_bar))]

extern crate unstable_feature;
use unstable_feature::{Foo, Bar, Moo};

/// To use impls gated by both `feat_foo` and `feat_moo`,
/// both features must be enabled.

fn main() {
    Bar::foo();
    Moo::foo();
    //[fail]~^ ERROR:use of unstable library feature `feat_moo` [E0658]
}
