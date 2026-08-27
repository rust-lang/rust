//@ aux-build:stable-glob-source.rs
//@ aux-build:unstable-glob-source.rs
//@ normalize-stderr: "(\n)\n$" -> "$1"

#![crate_type = "lib"]
#![feature(staged_api)]
#![deny(incompatible_reexport_stability)]
#![stable(feature = "reexport_test", since = "1.0.0")]

extern crate stable_glob_source;
extern crate unstable_glob_source;

// An unstable annotation does not match these stable items.
#[unstable(feature = "stable_glob_reexport", issue = "none")]
pub use stable_glob_source::*;
//~^ ERROR stability annotation on this re-export does not match the re-exported item

// Same feature and issue; `reason` does not matter.
#[unstable(feature = "unstable_glob_source", issue = "none")]
pub use unstable_glob_source::*;
