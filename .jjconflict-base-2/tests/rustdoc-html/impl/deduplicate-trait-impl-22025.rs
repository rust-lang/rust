//@ aux-build:issue-22025.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/22025
#![crate_name="issue_22025"]

extern crate issue_22025;

pub use issue_22025::foo::{Foo, Bar};
