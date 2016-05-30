#![feature(custom_attribute, test)]
#![feature(rustc_private)]
#![allow(unused_attributes)]

extern crate test;
use test::Bencher;

mod fibonacci_helper;

#[bench]
fn fib(bencher: &mut Bencher) {
    bencher.iter(|| {
        fibonacci_helper::main();
    })
}

mod miri_helper;

#[bench]
fn fib_miri(bencher: &mut Bencher) {
    miri_helper::run("fibonacci_helper", bencher);
}

mod fibonacci_helper_iterative;

#[bench]
fn fib_iter(bencher: &mut Bencher) {
    bencher.iter(|| {
        fibonacci_helper_iterative::main();
    })
}

#[bench]
fn fib_iter_miri(bencher: &mut Bencher) {
    miri_helper::run("fibonacci_helper_iterative", bencher);
}
