//@ run-pass
//@ aux-build:issue-41053.rs

#![allow(non_local_definitions)]

pub trait Trait { fn foo(&self) {} }

pub struct Foo;

impl Iterator for Foo {
    type Item = Box<dyn Trait>;
    fn next(&mut self) -> Option<Box<dyn Trait>> {
        extern crate issue_41053;
        impl crate::Trait for issue_41053::Test {
            fn foo(&self) {}
        }
        Some(Box::new(issue_41053::Test))
    }
}

fn main() {
    Foo.next().unwrap().foo();
}
