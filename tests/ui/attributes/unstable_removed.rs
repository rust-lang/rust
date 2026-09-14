//@ aux-build:unstable_removed_feature.rs

#![feature(old_feature)]
//~^ ERROR: feature `old_feature` has been removed

#![feature(concat_idents)]
//~^ ERROR: feature `concat_idents` has been removed

#![unstable_removed(
//~^ ERROR: stability attributes may not be used outside of the standard library
    feature = "old_feature",
    reason = "a good one",
    link = "https://github.com/rust-lang/rust/issues/141617",
    since="1.92.0"
)]

extern crate unstable_removed_feature;

fn main() {}
