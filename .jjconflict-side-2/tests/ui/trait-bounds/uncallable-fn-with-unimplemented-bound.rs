//! Regression test for <https://github.com/rust-lang/rust/issues/37725>.
//@ compile-flags: -Zmir-opt-level=2
//@ build-pass

#![allow(dead_code)]
trait Foo {
    fn foo(&self);
}

fn foo<'a>(s: &'a mut ()) where &'a mut (): Foo {
    s.foo();
}
fn main() {}
