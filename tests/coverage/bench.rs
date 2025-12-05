#![feature(test)]
//@ edition: 2021
//@ compile-flags: --test

extern crate test;

#[bench]
fn my_bench(_b: &mut test::Bencher) {}
