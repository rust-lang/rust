//! Regression test for https://github.com/rust-lang/rust/issues/23073
//@ check-pass
#![feature(associated_type_defaults)]

trait Foo { type T; }

trait Bar {
    type Foo: Foo;
    type FooT = <<Self as Bar>::Foo>::T;
}

fn main() {}
