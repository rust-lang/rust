#![allow(unused, clippy::empty_line_after_doc_comments)]
#![warn(clippy::suspicious_doc_comments)]
//@no-rustfix
///! a
//~^ suspicious_doc_comments

///! b
/// c
///! d
pub fn foo() {}

///! a
//~^ suspicious_doc_comments

///! b
/// c
///! d
use std::mem;

fn main() {}
