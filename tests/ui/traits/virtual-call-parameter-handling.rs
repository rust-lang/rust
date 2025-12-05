//! This test checks the correct parameter handling during virtual method calls
//! through a `dyn Trait` object.
//!
//! Regression test for: <https://github.com/rust-lang/rust/issues/137646>

//@ run-pass

use std::hint::black_box;

type T = (i32, i32, i32);

pub trait Trait {
    fn m(&self, _: T, _: T) {}
}

impl Trait for () {
    fn m(&self, mut _v1: T, v2: T) {
        _v1 = (0, 0, 0);
        check(v2);
    }
}

pub fn run_1(trait_: &dyn Trait) {
    let v1 = (1, 1, 1);
    let v2 = (1, 1, 1);
    trait_.m(v1, v2);
}

pub fn run_2(trait_: &dyn Trait) {
    let v1 = (1, 1, 1);
    let v2 = (1, 1, 1);
    trait_.m(v1, v2);
    check(v1);
    check(v2);
}

#[inline(never)]
fn check(v: T) {
    assert_eq!(v, (1, 1, 1));
}

fn main() {
    black_box(run_1 as fn(&dyn Trait));
    black_box(run_2 as fn(&dyn Trait));
    run_1(&());
    run_2(&());
}
