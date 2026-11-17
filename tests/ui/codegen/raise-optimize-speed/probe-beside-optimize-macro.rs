//! The items that a probe of `-Z raise-optimize-speed` adds to a crate carry no attributes in
//! their source. So a crate that has attribute macros named `optimize` and `used` in scope at
//! its root compiles with a probe as it does without, and so does the crate that defines them,
//! which is built with a probe too.

//@ build-pass
//@ edition: 2021
//@ proc-macro: optimize_macro.rs
//@ revisions: added_unused added
//@ compile-flags: -Copt-level=0
//@[added_unused] compile-flags: -Zraise-optimize-speed=modules,probe=added-unused
//@[added] compile-flags: -Zraise-optimize-speed=modules,probe=added

#![crate_type = "lib"]

pub use optimize_macro::{optimize, used};
