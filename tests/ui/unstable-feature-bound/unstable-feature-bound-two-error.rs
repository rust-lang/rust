//@ aux-build:unstable_feature.rs
extern crate unstable_feature;
use unstable_feature::{Foo, Bar, Moo};

// FIXME: both `feat_bar` and `feat_moo` are needed to pass this test,
// but the diagnostic only will point out `feat_bar`.

fn main() {
    Bar::foo();
    //~^ ERROR: use of unstable library feature `feat_bar` [E0658]
    Moo::foo();
}
