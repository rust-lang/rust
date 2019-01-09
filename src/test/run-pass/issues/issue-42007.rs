// run-pass
#![allow(dead_code)]
// aux-build:issue_42007_s.rs

extern crate issue_42007_s;

enum I {
    E(issue_42007_s::E),
}

fn main() {}
