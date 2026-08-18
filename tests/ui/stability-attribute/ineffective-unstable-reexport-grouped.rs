//@ aux-build:lint-stability.rs
//@ normalize-stderr: "(\n)\n$" -> "$1"

#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "reexport_test", since = "1.0.0")]

extern crate lint_stability;

// Both targets are stable.
// This should produce exactly one error for the shared attribute.
#[unstable(feature = "grouped_stable", issue = "none")]
pub use lint_stability::{
    stable as grouped_stable_a, //~ ERROR `#[unstable]` does not make this re-exported path unstable
    stable_text as grouped_stable_b,
};

// The annotation is used by #94972 to allow importing the unstable member
// without enabling its feature in this crate.
#[unstable(feature = "grouped_mixed", issue = "none")]
pub use lint_stability::{
    stable as grouped_mixed_stable,
    unstable as grouped_mixed_unstable,
};

// Both targets are already unstable. This is the #94972-style case and
// should not warn.
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub use lint_stability::{
    unstable as grouped_unstable_a,
    unstable_text as grouped_unstable_b,
};
