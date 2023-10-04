// aux-build:issue-22025.rs
// ignore-cross-compile

#![crate_name="issue_22025"]

extern crate issue_22025;

pub use issue_22025::foo::{Foo, Bar};
