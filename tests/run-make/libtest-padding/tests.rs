#![feature(test)]
extern crate test;

#[test]
fn short_test_name() {}

#[test]
fn this_is_a_really_long_test_name() {}

#[bench]
fn short_bench_name(b: &mut test::Bencher) {}

#[bench]
fn this_is_a_really_long_bench_name(b: &mut test::Bencher) {}
