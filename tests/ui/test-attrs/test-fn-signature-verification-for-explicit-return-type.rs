//@ run-pass
//@ ignore-fuchsia Test must be run out-of-process

#![feature(test)]

//@ compile-flags: --test
extern crate test;

#[bench]
pub fn bench_explicit_return_type(_: &mut ::test::Bencher) -> () {}

#[test]
pub fn test_explicit_return_type() -> () {}
