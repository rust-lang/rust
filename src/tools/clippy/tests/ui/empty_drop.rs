//@run-rustfix
#![warn(clippy::empty_drop)]
#![allow(unused)]

// should cause an error
struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {}
}

// shouldn't cause an error
struct Bar;

impl Drop for Bar {
    fn drop(&mut self) {
        println!("dropping bar!");
    }
}

// should error
struct Baz;

impl Drop for Baz {
    fn drop(&mut self) {
        {}
    }
}

fn main() {}
