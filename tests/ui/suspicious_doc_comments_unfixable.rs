#![allow(unused)]
#![warn(clippy::suspicious_doc_comments)]
//@no-rustfix
///! a
//~^ ERROR: this is an outer doc comment and does not apply to the parent module or crate
//~| NOTE: `-D clippy::suspicious-doc-comments` implied by `-D warnings`
///! b
/// c
///! d
pub fn foo() {}

///! a
//~^ ERROR: this is an outer doc comment and does not apply to the parent module or crate
///! b
/// c
///! d
use std::mem;

fn main() {}
