//@ aux-build:lint-stability.rs
//@ aux-build:stable-glob-source.rs
//@ check-pass
//@ normalize-stderr: "(\n)\n$" -> "$1"

#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "reexport_test", since = "1.0.0")]

extern crate lint_stability;
extern crate stable_glob_source;

// Every item introduced by this glob is stable, so the unstable annotation
// cannot make the exported paths unstable.
#[unstable(feature = "stable_glob_reexport", issue = "none")]
//~^ WARN `#[unstable]` does not make this re-exported path unstable
pub use stable_glob_source::*;

// This glob contains unstable items, so #94972 makes the annotation
// relevant when checking the import itself.
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub use lint_stability::*;
