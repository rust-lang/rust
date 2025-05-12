//@ run-pass
//@ aux-build:issue-19293.rs

extern crate issue_19293;
use issue_19293::{Foo, MyEnum};

fn main() {
    MyEnum::Foo(Foo(5));
}
