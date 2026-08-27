//@ aux-build:lint-stability.rs
//@ normalize-stderr: "(\n)\n$" -> "$1"

#![crate_type = "lib"]
#![feature(staged_api)]
#![deny(incompatible_reexport_stability)]
#![stable(feature = "reexport_test", since = "1.0.0")]

extern crate lint_stability;

// An unstable annotation does not match these stable items.
#[unstable(feature = "grouped_stable", issue = "none")]
pub use lint_stability::{
    stable as grouped_stable_a,
    stable_text as grouped_stable_b,
};
//~^^^ ERROR stability annotation on this re-export does not match the re-exported item

// The annotation must match every item.
#[unstable(feature = "grouped_mixed", issue = "none")]
pub use lint_stability::{
    stable as grouped_mixed_stable,
    unstable as grouped_mixed_unstable,
};
//~^^^ ERROR stability annotation on this re-export does not match the re-exported item

// Make sure we point at the later mismatch.
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub use lint_stability::{
    unstable as grouped_matching_first,
    stable as grouped_mismatching_second,
    //~^ ERROR stability annotation on this re-export does not match the re-exported item
};

// Same feature and issue; `reason` does not matter.
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub use lint_stability::{
    unstable as grouped_unstable_a,
    unstable_text as grouped_unstable_b,
};
