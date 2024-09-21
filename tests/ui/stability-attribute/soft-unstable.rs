// Test handling of soft-unstable items dependent on multiple features.
//@ aux-build:soft-unstable.rs
//@ revisions: all none
//@ [all]check-pass

#![cfg_attr(all, feature(a, b, c, d))]

extern crate soft_unstable;

fn main() {
    soft_unstable::mac!();
    //[none]~^ ERROR use of unstable library features `a` and `b`: reason [soft_unstable]
    //[none]~| WARNING this was previously accepted by the compiler but is being phased out
    soft_unstable::something();
    //[none]~^ ERROR use of unstable library features `c` and `d`: reason [soft_unstable]
    //[none]~| WARNING this was previously accepted by the compiler but is being phased out
}
