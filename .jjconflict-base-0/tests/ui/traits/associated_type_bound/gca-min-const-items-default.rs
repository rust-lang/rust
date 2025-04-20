//! Regression test for <https://github.com/rust-lang/rust/issues/156293>
//@ check-pass

#![feature(gca_min_const_items)]

trait Bar<const N: usize = const { 1 + 1 }> {}

trait Foo {
    type AssocB: Bar;
}

fn main() {}
