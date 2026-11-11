//@ run-pass
// Test that region inference correctly links up the regions when a
// `ref` borrow occurs inside a fn argument.


#![allow(dead_code)]

fn with<'a, F>(_: F) where F: FnOnce(&'a Vec<isize>) -> &'a Vec<isize> { }

fn foo() {
    with(|&ref ints| ints);
}

fn main() { }
