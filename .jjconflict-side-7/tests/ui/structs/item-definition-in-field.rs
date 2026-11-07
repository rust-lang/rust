//! Regression test for <https://github.com/rust-lang/rust/issues/29276>.
//! Item definitions in fields ICE'd.
//@ check-pass

#![allow(dead_code)]
struct S([u8; { struct Z; 0 }]);

fn main() {}
