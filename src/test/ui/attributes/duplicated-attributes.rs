// Test that, if an item is annotated with a builtin attribute more than once, a warning is
// emitted.
// Tests https://github.com/rust-lang/rust/issues/90979

// check-pass
// compile-flags: --test

#![feature(test)]
#![feature(cfg_eval)]

#[test]
#[test]
//~^ WARNING duplicated attribute
fn f() {}

// The following shouldn't trigger an error. The attribute is not duplicated.
#[test]
fn f2() {}

// The following shouldn't trigger an error either. The second attribute is not #[test].
#[test]
#[inline]
fn f3() {}

extern crate test;
use test::Bencher;

#[bench]
#[bench]
//~^ WARNING duplicated attribute
fn f4(_: &mut Bencher) {}

#[cfg_eval]
#[cfg_eval]
//~^ WARNING duplicated attribute
struct S;

#[cfg_eval]
struct S2;

fn main() {}
