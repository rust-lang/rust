//! Regression test for <https://github.com/rust-lang/rust/issues/18446>.
//! Tests that an inherent method on a trait object with existing default method
//! doesn't emit a duplicate definition error.

//@ check-pass
#![allow(dead_code)]

trait T {
    fn foo(&self) -> i32 { 0 }
}

impl<'a> dyn T + 'a {
    fn foo(&self) -> i32 { 1 }
}

fn main() {}
