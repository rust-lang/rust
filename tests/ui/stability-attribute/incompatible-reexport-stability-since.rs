//@ aux-build:lint-stability.rs
//@ normalize-stderr: "(\n)\n$" -> "$1"

#![crate_type = "lib"]
#![feature(staged_api)]
#![deny(incompatible_reexport_stability)]
#![stable(feature = "reexport_since_test", since = "1.0.0")]

extern crate lint_stability;

// Same feature, different `since`.
#[stable(feature = "rust1", since = "1.1.0")]
pub use lint_stability::stable as different_stable_since;
//~^ ERROR stability annotation on this re-export does not match the re-exported item
