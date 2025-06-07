//@ aux-build:unstable_feature.rs
//@ revisions: pass fail
//@[pass] check-pass

#![cfg_attr(pass, feature(feat_bar, feat_moo))]
#![cfg_attr(fail, feature(feat_bar))]

extern crate unstable_feature;
use unstable_feature::{Foo, Bar, Moo};

/// Both `feat_foo` and `feat_bar` are needed to use impl
/// gated by two different unstable feature bound.

fn main() {
    Bar::foo();
    Moo::foo();
    //[fail]~^ ERROR: type annotations needed: cannot satisfy `unstable feature: `feat_moo``
}
