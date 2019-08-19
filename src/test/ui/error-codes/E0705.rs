// build-pass (FIXME(62277): could be check-pass?)

// This is a stub feature that doesn't control anything, so to make tidy happy,
// gate-test-test_2018_feature

#![feature(test_2018_feature)]
//~^ WARN the feature `test_2018_feature` is included in the Rust 2018 edition
#![feature(rust_2018_preview)]

fn main() {}
