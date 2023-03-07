// run-pass
#![allow(dead_code)]
// aux-build:issue-42007-s.rs

extern crate issue_42007_s;

enum I {
    E(issue_42007_s::E),
}

fn main() {}
