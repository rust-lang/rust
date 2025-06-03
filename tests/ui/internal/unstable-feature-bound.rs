//@ aux-build:unstable_feature.rs

extern crate unstable_feature;
use unstable_feature::{Foo, Bar};

fn main() {
    Bar::foo(); //~ ERROR: type annotations needed: cannot satisfy `unstable feature: `feat_foo``
}
