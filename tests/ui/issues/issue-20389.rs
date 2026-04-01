//@ run-pass
#![allow(dead_code)]
//@ aux-build:issue-20389.rs


extern crate issue_20389;

struct Foo;

impl issue_20389::T for Foo {
    type C = ();
}

fn main() {}
