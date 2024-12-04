//@ proc-macro: issue-49482-macro-def.rs
#[macro_use]
extern crate issue_49482_macro_def;

pub use issue_49482_macro_def::*;

pub fn foo() {}
