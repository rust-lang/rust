//@ aux-build:lint-stability.rs
//@ normalize-stderr: "(\n)\n$" -> "$1"

#![crate_type = "lib"]
#![feature(staged_api)]
#![deny(ineffective_unstable_reexports)]
#![stable(feature = "reexport_test", since = "1.0.0")]

extern crate lint_stability;

// The annotation is ineffective when every re-exported target is stable.
#[unstable(feature = "grouped_stable", issue = "none")]
pub use lint_stability::{
    stable as grouped_stable_a,
    stable_text as grouped_stable_b,
};
//~^^^ ERROR `#[unstable]` does not make this re-exported path unstable

// The annotation is not wholly ineffective if any target is unstable.
#[unstable(feature = "grouped_mixed", issue = "none")]
pub use lint_stability::{
    stable as grouped_mixed_stable,
    unstable as grouped_mixed_unstable,
};

// Order must not matter.
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub use lint_stability::{
    unstable as grouped_unstable_first,
    stable as grouped_stable_second,
};

// All unstable targets are fine.
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub use lint_stability::{
    unstable as grouped_unstable_a,
    unstable_text as grouped_unstable_b,
};
