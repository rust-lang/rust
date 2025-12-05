//@ check-pass
//@ compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/213>.

use std::ops::Deref;

trait Trait: Deref<Target = [u8; { 1 + 1 }]> {}

fn main() {}
