//! Analysis of patterns, notably match exhaustiveness checking.

pub mod constructor;
pub mod errors;
pub mod pat;
pub mod usefulness;

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
