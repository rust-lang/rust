//! Make sure that incomplete features emit the `incomplete_features` lint.

// gate-test-test_incomplete_feature

//@ check-pass
//@ revisions: warn expect

#![cfg_attr(warn, warn(incomplete_features))]
#![cfg_attr(expect, expect(incomplete_features))]

#![feature(test_incomplete_feature)] //[warn]~ WARN the feature `test_incomplete_feature` is incomplete

fn main() {}
