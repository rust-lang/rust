//@ aux-build:lint-stability.rs
//@ normalize-stderr: "(\n)\n$" -> "$1"

#![crate_type = "lib"]
#![feature(staged_api)]
#![deny(incompatible_reexport_stability)]
#![stable(feature = "reexport_test", since = "1.0.0")]

extern crate core;
extern crate lint_stability;

// An unstable annotation cannot make a stable item unstable through a re-export.
#[unstable(feature = "reexport_test_unstable", issue = "none")]
pub use lint_stability::stable as supposedly_unstable;
//~^ ERROR stability annotation on this re-export does not match the re-exported item

// Repeating the target's stable metadata is fine.
#[stable(feature = "rust1", since = "1.0.0")]
pub use lint_stability::stable as matching_stable;

// A stable re-export must use the same stability feature.
#[stable(feature = "different_stable_feature", since = "1.0.0")]
pub use lint_stability::stable as different_stable_feature;
//~^ ERROR stability annotation on this re-export does not match the re-exported item

// Repeating the target's unstable feature and issue is fine.
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub use lint_stability::unstable as matching_unstable;

// An unstable re-export must use the same feature.
#[unstable(feature = "different_unstable_feature", issue = "none")]
pub use lint_stability::unstable as different_unstable_feature;
//~^ ERROR stability annotation on this re-export does not match the re-exported item

// An unstable re-export must use the same tracking issue.
#[unstable(feature = "unstable_test_feature", issue = "12345")]
pub use lint_stability::unstable as different_unstable_issue;
//~^ ERROR stability annotation on this re-export does not match the re-exported item

// Primitive re-exports have no DefId, but primitives themselves are stable.
#[unstable(feature = "primitive_reexport", issue = "none")]
pub use core::primitive::bool as supposedly_unstable_bool;
//~^ ERROR stability annotation on this re-export does not match the re-exported item
