//! Regression test for <https://github.com/rust-lang/rust/issues/156293>
//@ check-pass

#![feature(min_generic_const_args)]

trait Bar<const N: bool = false> {}

trait Foo {
    type AssocB: Bar;
}

fn main() {}
