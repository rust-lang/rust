//@ aux-build:stable-glob-source.rs
//@ aux-build:unstable-glob-source.rs
//@ normalize-stderr: "(\n)\n$" -> "$1"

#![crate_type = "lib"]
#![feature(staged_api)]
#![deny(ineffective_unstable_reexports)]
#![stable(feature = "reexport_test", since = "1.0.0")]

extern crate stable_glob_source;
extern crate unstable_glob_source;

// The unstable annotation is ineffective because every target is stable.
#[unstable(feature = "stable_glob_reexport", issue = "none")]
pub use stable_glob_source::*;
//~^ ERROR `#[unstable]` does not make this re-exported path unstable

// The annotation remains meaningful because these targets are unstable.
#[unstable(feature = "unstable_glob_source", issue = "none")]
pub use unstable_glob_source::*;
