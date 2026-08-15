//@ check-pass
//@ proc-macro: issue-79825.rs
#![feature(trait_alias)]

extern crate issue_79825;

#[issue_79825::assert_input]
trait Alias = Sized;

fn main() {}
