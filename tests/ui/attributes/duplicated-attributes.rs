// Test that, if an item is annotated with a builtin attribute more than once, a warning is
// emitted.
// Tests https://github.com/rust-lang/rust/issues/90979

//@ check-pass
//@ compile-flags: --test

#![feature(test)]
#![feature(cfg_eval)]
#![feature(custom_test_frameworks)]

#[test]
#[test]
//~^ WARNING multiple `test` attributes
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
//~^ WARNING multiple `bench` attributes
fn f4(_: &mut Bencher) {}

#[test]
#[test]
//~^ WARNING multiple `test` attributes
#[test]
//~^ WARNING multiple `test` attributes
fn f5() {}

#[test_case]
#[test_case]
//~^ WARNING multiple `test_case` attributes
struct T;

#[cfg_eval]
#[cfg_eval]
//~^ WARNING multiple `cfg_eval` attributes
struct S;

#[cfg_eval]
struct S2;

fn main() {}
