//@ aux-build:lint-stability.rs
//@ check-pass

#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "reexport_test", since = "1.0.0")]

extern crate lint_stability;

// An unstable annotation cannot currently make a stable item unstable
// through a re-export.
#[unstable(feature = "reexport_test_unstable", issue = "none")]
//~^ WARN `#[unstable]` does not make this re-exported path unstable
pub use lint_stability::stable as supposedly_unstable;

// This is intentional: #94972 allows an unstable upstream item to be
// re-exported without enabling its feature in this crate.
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub use lint_stability::unstable as still_unstable;
