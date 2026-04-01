//@ check-pass
#![allow(dead_code)]
// Test that methods in trait impls should override default methods.

trait T {
    fn foo(&self) -> i32 { 0 }
}

impl<'a> dyn T + 'a {
    fn foo(&self) -> i32 { 1 }
}

fn main() {}
