//! Test error for explicit destructor method calls via UFCS

//@ run-rustfix

#![allow(dropping_references)]

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn main() {
    Drop::drop(&mut Foo) //~ ERROR explicit use of destructor method
}
