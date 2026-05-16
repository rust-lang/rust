//@ run-pass
//@ aux-build:issue-31702-1.rs
//@ aux-build:issue-31702-2.rs

#![allow(unused_unconstructable_pub_structs)]
// this test is actually entirely in the linked library crates

extern crate issue_31702_1;
extern crate issue_31702_2;

fn main() {}
