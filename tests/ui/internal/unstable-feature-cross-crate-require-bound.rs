//@ aux-build:unstable_feature.rs

extern crate unstable_feature;
use unstable_feature::{Foo, Bar};

/// #[feature(..)] is required to use unstable impl.
/// FIXME: write check-pass variant

fn main() {
    Bar::foo();
    //~^ ERROR: cannot satisfy `unstable feature: `feat_bar``
}
