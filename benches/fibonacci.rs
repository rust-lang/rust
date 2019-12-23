#![feature(test, rustc_private)]

extern crate test;
use crate::test::Bencher;
mod helpers;
use crate::helpers::*;

#[bench]
fn fib(bencher: &mut Bencher) {
    bencher.iter(|| fibonacci_helper::main())
}

#[bench]
fn fib_miri(bencher: &mut Bencher) {
    miri_helper::run("fibonacci_helper", bencher);
}

#[bench]
fn fib_iter(bencher: &mut Bencher) {
    bencher.iter(|| fibonacci_helper_iterative::main())
}

#[bench]
fn fib_iter_miri(bencher: &mut Bencher) {
    miri_helper::run("fibonacci_helper_iterative", bencher);
}
