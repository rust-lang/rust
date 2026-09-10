//@ aux-build:lint-stability.rs
//@ aux-build:non-staged-reexport-source.rs
//@ normalize-stderr: "(\n)\n$" -> "$1"

#![crate_type = "lib"]
#![feature(staged_api)]
#![deny(ineffective_unstable_reexports)]
#![stable(feature = "reexport_test", since = "1.0.0")]

extern crate core;
extern crate lint_stability;
extern crate non_staged_reexport_source;

// `#[unstable]` cannot make an otherwise stable re-exported path unstable.
#[unstable(feature = "reexport_test_unstable", issue = "none")]
pub use lint_stability::stable as supposedly_unstable;
//~^ ERROR `#[unstable]` does not make this re-exported path unstable

// Stable re-exports are outside the scope of this lint.
#[stable(feature = "rust1", since = "1.0.0")]
pub use lint_stability::stable as matching_stable;

#[stable(feature = "different_stable_feature", since = "1.0.0")]
pub use lint_stability::stable as different_stable_feature;

// `#[unstable]` remains meaningful when the target is itself unstable.
// The feature and issue do not need to match for this lint.
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub use lint_stability::unstable as matching_unstable;

#[unstable(feature = "different_unstable_feature", issue = "none")]
pub use lint_stability::unstable as different_unstable_feature;

#[unstable(feature = "unstable_test_feature", issue = "12345")]
pub use lint_stability::unstable as different_unstable_issue;

// Items from crates without staged API metadata are effectively stable.
#[unstable(feature = "non_staged_reexport", issue = "none")]
pub use non_staged_reexport_source::stable as supposedly_unstable_external;
//~^ ERROR `#[unstable]` does not make this re-exported path unstable

// Primitives have no DefId, but they are stable.
#[unstable(feature = "primitive_reexport", issue = "none")]
pub use core::primitive::bool as supposedly_unstable_bool;
//~^ ERROR `#[unstable]` does not make this re-exported path unstable
