#![allow(unused)]
#![warn(clippy::suspicious_doc_comments)]
//@no-rustfix
///! a
///! b
/// c
///! d
pub fn foo() {}

///! a
///! b
/// c
///! d
use std::mem;

fn main() {}
