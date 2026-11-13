//! Regression test for #13560. Previously, it was possible to
//! trigger an assert in crate numbering if a series of crates
//! being loaded included a "syntax-only" extern crate.
//! But it appears we don't mess with crate numbering for
//! `#[no_link]` crates anymore, so this test doesn't seem
//! to test anything now.

//@ run-pass
//@ needs-crate-type: dylib
//@ aux-build:empty-crate-1.rs
//@ aux-build:empty-crate-2.rs
//@ aux-build:no_link-crate.rs

extern crate empty_crate_2 as t2;
extern crate no_link_crate as t3;

fn main() {}
