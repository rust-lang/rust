// aux-build:issue-34274.rs
// build-aux-docs
// ignore-cross-compile

#![crate_name = "foo"]

extern crate issue_34274;

// @has foo/fn.extern_c_fn.html '//a/@href' '../src/issue_34274/issue-34274.rs.html#2'
pub use issue_34274::extern_c_fn;
