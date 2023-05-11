// run-pass
#![allow(dead_code)]
// aux-build:issue-20389.rs

// pretty-expanded FIXME #23616

extern crate issue_20389;

struct Foo;

impl issue_20389::T for Foo {
    type C = ();
}

fn main() {}
