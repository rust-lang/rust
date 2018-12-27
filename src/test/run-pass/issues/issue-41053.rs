// run-pass
// aux-build:issue_41053.rs

pub trait Trait { fn foo(&self) {} }

pub struct Foo;

impl Iterator for Foo {
    type Item = Box<Trait>;
    fn next(&mut self) -> Option<Box<Trait>> {
        extern crate issue_41053;
        impl ::Trait for issue_41053::Test {
            fn foo(&self) {}
        }
        Some(Box::new(issue_41053::Test))
    }
}

fn main() {
    Foo.next().unwrap().foo();
}
