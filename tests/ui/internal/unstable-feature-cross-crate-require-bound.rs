//@ aux-build:unstable_feature.rs
//@ revisions: pass fail
//@[pass] check-pass

#![cfg_attr(pass, feature(feat_bar))]
extern crate unstable_feature;
use unstable_feature::{Foo, Bar};

/// #[feature(..)] is required to use unstable impl.

fn main() {
    Bar::foo();
    //[fail]~^ ERROR: cannot satisfy `unstable feature: `feat_bar``
}
