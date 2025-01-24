//@ run-pass

#![allow(dead_code, unused_macros)]
//@ proc-macro: issue-39889.rs

extern crate issue_39889;
use issue_39889::Issue39889;

#[derive(Issue39889)]
struct S;

fn main() {}
