// This tests the interaction of feature staging and supertrait item shadowing.
// When a feature is *off*, then we should not consider unstable methods for probing.
// When a feature is *on*, then we follow the normal supertrait item shadowing rules:
//   - When supertrait item shadowing is disabled, this is a clash.
//   - When supertrait item shadowing is enabled, we pick subtraits.

//@ aux-build: shadowed_stability.rs
//@ revisions: off_normal on_normal off_shadowing on_shadowing
//@[off_normal] run-pass
//@[on_normal] check-fail
//@[off_shadowing] run-pass
//@[on_shadowing] run-pass
//@ check-run-results

#![allow(dead_code, unused_features, unused_imports)]
#![cfg_attr(on_shadowing, feature(downstream))]
#![cfg_attr(on_normal, feature(downstream))]
#![cfg_attr(off_shadowing, feature(supertrait_item_shadowing))]
#![cfg_attr(on_shadowing, feature(supertrait_item_shadowing))]

extern crate shadowed_stability;
use shadowed_stability::*;

fn main() {
    ().hello();
    //[off_normal,off_shadowing]~^ WARN a method with this name may be added
    //[off_normal,off_shadowing]~| WARN once this associated item is added
    //[on_normal]~^^^ ERROR multiple applicable items in scope
}
