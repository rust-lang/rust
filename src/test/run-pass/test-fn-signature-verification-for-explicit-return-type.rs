#![feature(test)]

// compile-flags: --test
extern crate test;

#[bench]
pub fn bench_explicit_return_type(_: &mut ::test::Bencher) -> () {}

#[test]
pub fn test_explicit_return_type() -> () {}
