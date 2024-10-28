//! Checks that multiple stability attributes are used correctly together

#![feature(staged_api)]

#![stable(feature = "stable_test_feature", since = "1.0.0")]

#[unstable(feature = "a", issue = "none", reason = "reason 1")]
#[unstable(feature = "b", issue = "none", reason = "reason 2")] //~ ERROR multiple reasons provided for unstability
fn f1() { }

#[unstable(feature = "a", issue = "none", reason = "reason 1")]
#[unstable(feature = "b", issue = "none", reason = "reason 2")] //~ ERROR multiple reasons provided for unstability
#[unstable(feature = "c", issue = "none", reason = "reason 3")] //~ ERROR multiple reasons provided for unstability
fn f2() { }

#[unstable(feature = "a", issue = "none")] //~ ERROR `soft` must be present on either none or all of an item's `unstable` attributes
#[unstable(feature = "b", issue = "none", soft)]
fn f3() { }

#[unstable(feature = "a", issue = "none", soft)] //~ ERROR `soft` must be present on either none or all of an item's `unstable` attributes
#[unstable(feature = "b", issue = "none")]
fn f4() { }

fn main() { }
